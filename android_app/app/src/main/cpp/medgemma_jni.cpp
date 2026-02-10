/**
 * MedGemma JNI Bridge
 *
 * Minimal JNI wrapper around llama.cpp for running MedGemma GGUF models
 * on Android. Provides: init, load, generate (streaming), benchmark,
 * systemInfo, unload.
 *
 * Streaming: nativeGenerate writes tokens into a shared buffer.
 *   Kotlin polls nativeGetPartialResult() every ~200ms.
 *   nativeStopGeneration() sets a flag to abort early.
 *
 * JNI class: com.medgemma.edge.inference.MedGemmaInference
 */

#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <chrono>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "ggml.h"

#define TAG "MedGemma"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGw(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Global state (single model at a time)
// ---------------------------------------------------------------------------
static llama_model   * g_model   = nullptr;
static llama_context * g_context = nullptr;

// ── Streaming state ─────────────────────────────────────────────────────────
static std::mutex       g_result_mutex;
static std::string      g_partial_result;       // accumulates generated text
static std::atomic<bool> g_stop_flag{false};     // set true to abort generation
static std::atomic<int>  g_tokens_generated{0};  // token counter
static std::atomic<bool> g_is_generating{false};  // true while generate() runs
static std::atomic<float> g_tok_per_sec{0.0f};   // real-time speed

static void log_callback(ggml_log_level level, const char * text, void *) {
    int prio = ANDROID_LOG_DEBUG;
    switch (level) {
        case GGML_LOG_LEVEL_INFO:  prio = ANDROID_LOG_INFO;  break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        default: break;
    }
    __android_log_print(prio, TAG, "%s", text);
}

// ---------------------------------------------------------------------------
// JNI exports
// ---------------------------------------------------------------------------
extern "C" {

/**
 * Initialize llama.cpp backend.
 * Must be called once before any other native method.
 */
JNIEXPORT void JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeInit(
        JNIEnv * env, jobject, jstring jNativeLibDir) {

    llama_log_set(log_callback, nullptr);

    // Try loading dynamic backends from app's native lib directory
    const char * libdir = env->GetStringUTFChars(jNativeLibDir, nullptr);
    LOGi("Init: native lib dir = %s", libdir);
    ggml_backend_load_all_from_path(libdir);
    env->ReleaseStringUTFChars(jNativeLibDir, libdir);

    llama_backend_init();
    LOGi("Backend initialized");
}

/**
 * Load a GGUF model from the given file path.
 * Uses mmap + mlock to map the model AND pin it in physical RAM.
 * Creates a context with 1024 token window.
 * Returns 0 on success, non-zero on failure.
 */
JNIEXPORT jint JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeLoadModel(
        JNIEnv * env, jobject, jstring jModelPath) {

    // Unload previous model if any
    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model)   { llama_model_free(g_model); g_model = nullptr; }

    const char * path = env->GetStringUTFChars(jModelPath, nullptr);
    LOGi("Loading model: %s", path);

    // Log system info BEFORE model load
    LOGi("System info: %s", llama_print_system_info());
    LOGi("mmap supported: %d, mlock supported: %d",
         llama_supports_mmap(), llama_supports_mlock());

    llama_model_params model_params = llama_model_default_params();
    // mmap=false forces a sequential read into malloc'd RAM.
    // On Android, mlock requires CAP_IPC_LOCK (RLIMIT_MEMLOCK=64KB for apps),
    // so mmap+mlock silently degrades to demand-paging from flash at ~0.7 GB/s.
    // With mmap=false, the full 2.3 GB is read upfront (~25-30s) but then all
    // weights are in anonymous pages that the kernel won't page out to flash.
    model_params.use_mmap  = false;
    model_params.use_mlock = false;
    LOGi("Loading with mmap=false (full RAM load)...");

    auto t_load_start = std::chrono::high_resolution_clock::now();
    g_model = llama_model_load_from_file(path, model_params);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t_load_end - t_load_start).count();

    env->ReleaseStringUTFChars(jModelPath, path);

    if (!g_model) {
        LOGe("Failed to load model");
        return 1;
    }
    LOGi("Model loaded into RAM in %.1f seconds", load_s);

    // Create inference context — try different thread counts
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx             = 512;   // minimal context for demo
    ctx_params.n_batch           = 512;
    ctx_params.n_threads         = 4;
    ctx_params.n_threads_batch   = 4;
    ctx_params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // safer fallback

    LOGi("Creating context: n_ctx=512, threads=4, flash_attn=disabled");

    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) {
        LOGe("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return 2;
    }

    LOGi("Model and context created successfully");
    return 0;
}

/**
 * Generate text given a prompt string.
 * Writes tokens into g_partial_result as they are generated.
 * Kotlin should poll nativeGetPartialResult() for streaming updates.
 * Returns the final generated text (not including the prompt).
 */
JNIEXPORT jstring JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeGenerate(
        JNIEnv * env, jobject, jstring jPrompt, jint maxTokens) {

    if (!g_model || !g_context) {
        return env->NewStringUTF("[Error: model not loaded]");
    }

    // Reset streaming state
    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        g_partial_result.clear();
    }
    g_stop_flag.store(false);
    g_tokens_generated.store(0);
    g_is_generating.store(true);
    g_tok_per_sec.store(0.0f);

    const char * prompt_cstr = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(prompt_cstr);
    env->ReleaseStringUTFChars(jPrompt, prompt_cstr);

    // Tokenize
    std::vector<llama_token> tokens = common_tokenize(g_context, prompt, true);
    LOGi("Prompt tokens: %zu", tokens.size());

    if (tokens.empty()) {
        g_is_generating.store(false);
        return env->NewStringUTF("[Error: empty prompt after tokenization]");
    }

    // Clear KV cache
    llama_memory_clear(llama_get_memory(g_context), false);

    // Evaluate prompt
    const int n_prompt = (int) tokens.size();
    LOGi("Evaluating %d prompt tokens...", n_prompt);

    auto t_prompt_start = std::chrono::high_resolution_clock::now();

    llama_batch batch = llama_batch_init(std::max(n_prompt, 512), 0, 1);

    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(g_context, batch) != 0) {
        llama_batch_free(batch);
        g_is_generating.store(false);
        return env->NewStringUTF("[Error: failed to evaluate prompt]");
    }

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();
    double pp_speed = (prompt_ms > 0) ? (n_prompt / (prompt_ms / 1000.0)) : 0;
    LOGi("Prompt eval: %.0f ms (%.1f tok/s)", prompt_ms, pp_speed);

    // Setup sampler (low temperature for medical responses)
    common_params_sampling sparams;
    sparams.temp = 0.3f;
    common_sampler * sampler = common_sampler_init(g_model, sparams);

    // Generate tokens (streaming into g_partial_result)
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    int pos = n_prompt;
    int n_gen = 0;

    auto t_gen_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < maxTokens; i++) {
        // Check stop flag
        if (g_stop_flag.load()) {
            LOGi("Generation stopped by user after %d tokens", n_gen);
            break;
        }

        llama_token new_token = common_sampler_sample(sampler, g_context, -1);
        common_sampler_accept(sampler, new_token, true);

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOGi("EOS reached after %d tokens", n_gen);
            break;
        }

        std::string piece = common_token_to_piece(g_context, new_token);
        n_gen++;

        // Update streaming state
        {
            std::lock_guard<std::mutex> lock(g_result_mutex);
            g_partial_result += piece;
        }
        g_tokens_generated.store(n_gen);

        // Update speed every 4 tokens
        if (n_gen % 4 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_gen_start).count();
            if (elapsed > 0) {
                g_tok_per_sec.store((float)(n_gen / elapsed));
            }
        }

        // Log every 16 tokens
        if (n_gen % 16 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_gen_start).count();
            double speed = (elapsed > 0) ? (n_gen / elapsed) : 0;
            LOGi("Generated %d/%d tokens (%.1f tok/s)", n_gen, (int)maxTokens, speed);
        }

        // Decode the new token
        common_batch_clear(batch);
        common_batch_add(batch, new_token, pos++, {0}, true);
        if (llama_decode(g_context, batch) != 0) {
            LOGe("Failed to decode generated token %d", n_gen);
            break;
        }
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    double tg_speed = (gen_ms > 0) ? (n_gen / (gen_ms / 1000.0)) : 0;

    LOGi("Generation done: %d tokens in %.0f ms (%.2f tok/s)", n_gen, gen_ms, tg_speed);
    LOGi("Prompt: %.0f ms (%.1f tok/s), Generation: %.0f ms (%.1f tok/s)",
         prompt_ms, pp_speed, gen_ms, tg_speed);

    common_sampler_free(sampler);
    llama_batch_free(batch);

    g_is_generating.store(false);

    // Build final result with stats header
    std::string final_result;
    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        final_result = g_partial_result;
    }

    char stats[512];
    snprintf(stats, sizeof(stats),
        "[pp: %d tok in %.0fms = %.1f tok/s | gen: %d tok in %.0fms = %.1f tok/s]",
        n_prompt, prompt_ms, pp_speed, n_gen, gen_ms, tg_speed);

    std::string output = std::string(stats) + "\n\n" + final_result;
    return env->NewStringUTF(output.c_str());
}

/**
 * Get the current partial generation result (for streaming UI updates).
 * Returns: "TOKENS|SPEED|IS_GENERATING|TEXT"
 */
JNIEXPORT jstring JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeGetPartialResult(
        JNIEnv * env, jobject) {
    std::string text;
    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        text = g_partial_result;
    }
    int tokens = g_tokens_generated.load();
    float speed = g_tok_per_sec.load();
    bool generating = g_is_generating.load();

    // Format: tokens|speed|is_generating|text
    char header[128];
    snprintf(header, sizeof(header), "%d|%.1f|%d|", tokens, speed, generating ? 1 : 0);
    std::string result = std::string(header) + text;
    return env->NewStringUTF(result.c_str());
}

/**
 * Signal the generation loop to stop.
 */
JNIEXPORT void JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeStopGeneration(
        JNIEnv *, jobject) {
    LOGi("Stop requested");
    g_stop_flag.store(true);
}

/**
 * Run prompt-processing and token-generation benchmarks.
 * Returns formatted results string.
 */
JNIEXPORT jstring JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeBench(
        JNIEnv * env, jobject, jint pp, jint tg, jint reps) {

    if (!g_model) {
        return env->NewStringUTF("Error: model not loaded");
    }

    // Create a dedicated context for benchmarking
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx             = pp + tg + 64;
    ctx_params.n_batch           = std::max((int)pp, 512);
    ctx_params.n_threads         = 4;
    ctx_params.n_threads_batch   = 4;
    ctx_params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context * bench_ctx = llama_init_from_model(g_model, ctx_params);
    if (!bench_ctx) {
        return env->NewStringUTF("Error: failed to create bench context");
    }

    llama_batch batch = llama_batch_init(std::max((int)pp, 512), 0, 1);
    double pp_sum = 0.0, tg_sum = 0.0;

    for (int r = 0; r < reps; r++) {
        // ── Prompt processing benchmark ──
        common_batch_clear(batch);
        for (int i = 0; i < pp; i++) {
            common_batch_add(batch, 0, i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(bench_ctx), false);

        const int64_t t_pp_start = ggml_time_us();
        if (llama_decode(bench_ctx, batch) != 0) {
            LOGe("llama_decode failed during pp bench");
            break;
        }
        const int64_t t_pp_end = ggml_time_us();

        // ── Token generation benchmark ──
        llama_memory_clear(llama_get_memory(bench_ctx), false);
        const int64_t t_tg_start = ggml_time_us();
        for (int i = 0; i < tg; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, 0, i, {0}, true);
            if (llama_decode(bench_ctx, batch) != 0) {
                LOGe("llama_decode failed during tg bench at token %d", i);
                break;
            }
        }
        const int64_t t_tg_end = ggml_time_us();

        const double t_pp = (t_pp_end - t_pp_start) / 1000000.0;
        const double t_tg = (t_tg_end - t_tg_start) / 1000000.0;

        const double pp_speed = (t_pp > 0) ? pp / t_pp : 0;
        const double tg_speed = (t_tg > 0) ? tg / t_tg : 0;

        pp_sum += pp_speed;
        tg_sum += tg_speed;
        LOGi("Rep %d/%d: pp=%.2f tok/s, tg=%.2f tok/s", r+1, reps, pp_speed, tg_speed);
    }

    llama_batch_free(batch);
    llama_free(bench_ctx);

    pp_sum /= reps;
    tg_sum /= reps;

    char buf[1024];
    snprintf(buf, sizeof(buf),
        "=== MEDGEMMA BENCHMARK ===\n"
        "Prompt Processing: %.2f tok/s (%d tokens)\n"
        "Token Generation:  %.2f tok/s (%d tokens)\n"
        "Repetitions: %d\n"
        "\n"
        "Device: %s\n"
        "System: %s",
        pp_sum, (int)pp,
        tg_sum, (int)tg,
        (int)reps,
        llama_print_system_info(),
        llama_print_system_info());

    return env->NewStringUTF(buf);
}

/**
 * Return llama.cpp system info string (CPU features, etc.)
 */
JNIEXPORT jstring JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeSystemInfo(
        JNIEnv * env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

/**
 * Unload the current model and free resources.
 */
JNIEXPORT void JNICALL
Java_com_medgemma_edge_inference_MedGemmaInference_nativeUnload(
        JNIEnv *, jobject) {
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    LOGi("Model unloaded");
}

} // extern "C"
