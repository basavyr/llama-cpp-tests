#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

FILE *log_file;

struct ModelParams
{
    int n_gpu_layers;
    std::string model_path;
    llama_model_params model_params;
};

struct ContextInit
{
    llama_context_params params;
    llama_context *ctx;
};

ContextInit get_context(llama_model *model, int n_input_tokens, int max_output_tokens)
{
    // llama model should be passed as a pointer, since it is an opaque pointer and it cannot be used via &
    // source: https://stackoverflow.com/questions/7058339/when-should-i-use-pointers-instead-of-references-in-api-design/7058373#7058373
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_input_tokens + max_output_tokens - 1;
    params.n_batch = n_input_tokens;
    params.no_perf = false;

    llama_context *ctx = llama_init_from_model(model, params);

    return ContextInit{params, ctx};
}

void print_token(int token_id, const llama_vocab *vocab)
{
    char buffer[128];
    int token_to_piece = llama_token_to_piece(vocab, token_id, buffer, sizeof(buffer), 0, true);

    std::string token_as_str(buffer, token_to_piece);
    printf("Token: %i -> Str: %s\n", token_id, token_as_str.c_str());
}

llama_sampler *get_sampler()
{
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    return sampler;
}

struct llama_model_and_vocab
{
    llama_model *model;
    const llama_vocab *vocab;
};

llama_model_and_vocab get_model_and_vocab(const std::string &model_path)
{
    ModelParams model_config = ModelParams();
    model_config.model_path = model_path;
    model_config.n_gpu_layers = 999;
    model_config.model_params = llama_model_default_params();

    // model and vocab
    llama_model *model = llama_model_load_from_file(model_config.model_path.c_str(), model_config.model_params);
    const llama_vocab *vocab = llama_model_get_vocab(model);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        exit(EXIT_FAILURE);
    }

    return llama_model_and_vocab{model, vocab};
}

struct llama_tokenizer
{
    const int n_input_tokens;
    std::vector<llama_token> prompt_tokens;
    ContextInit context;
};

llama_tokenizer tokenize_prompt(llama_model_and_vocab *model_and_vocab, const std::string &input_prompt, const int max_output_tokens)
{
    /*
    llama_tokenize args:
    const struct llama_vocab * vocab,
                const char * text,
                    int32_t   text_len,
                llama_token * tokens,
                    int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special)
    */
    const int n_input_tokens = -llama_tokenize(model_and_vocab->vocab, input_prompt.c_str(), input_prompt.size(), NULL, 0, true, true);
    ContextInit context = get_context(model_and_vocab->model, n_input_tokens, max_output_tokens);
    // tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_input_tokens);
    int n_tokens_max = prompt_tokens.size();

    if (llama_tokenize(model_and_vocab->vocab, input_prompt.c_str(), input_prompt.size(), prompt_tokens.data(), n_tokens_max, true, true) < 0)
    {
        fprintf(stderr, "failed to tokenize the prompt\n");
        exit(EXIT_FAILURE);
    }

    return llama_tokenizer{n_input_tokens, prompt_tokens, context};
}

void batch_decode(
    llama_model *model,
    const llama_vocab *vocab,
    std::vector<llama_token> *prompt_tokens,
    ContextInit *context,
    int n_input_tokens,
    const int max_output_tokens,
    bool generate_logs)
{
    const auto t_start = ggml_time_us();
    /*
    llama_batch args:
        int32_t         n_tokens;
        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;
    */
    llama_batch batch = llama_batch_get_one(prompt_tokens->data(), prompt_tokens->size()); // use `->` instead of `.` when the object is passed as a pointer

    // for decoder it should be -1
    llama_token decoder_start_token_id = llama_model_decoder_start_token(model);

    if (decoder_start_token_id == LLAMA_TOKEN_NULL)
    {
        decoder_start_token_id = llama_vocab_bos(vocab);
    }

    llama_sampler *sampler = get_sampler();

    int decode_steps = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_input_tokens + max_output_tokens;)
    {
        llama_decode(context->ctx, batch); // use `->` instead of `.` when the object is passed as a pointer
        n_pos += batch.n_tokens;

        // sampling the next token
        {
            new_token_id = llama_sampler_sample(sampler, context->ctx, -1);
            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id))
            {
                break;
            }

            char buffer[128];
            int current_token_id = llama_token_to_piece(vocab, new_token_id, buffer, sizeof(buffer), 0, true); // this must be positive
            std::string token_string(buffer, current_token_id);
            printf("%s", token_string.c_str());
            fflush(stdout);

            batch = llama_batch_get_one(&new_token_id, 1);

            decode_steps += 1;
        }
    }
    const auto t_end = ggml_time_us();
    auto duration_s = (t_end - t_start) / 1000000.0f;
    auto tokens_per_s = decode_steps / duration_s;

    if (generate_logs == true)
    {
        fprintf(log_file, "%i, %f, %f\n", decode_steps, duration_s, tokens_per_s);
    }

    printf("\n\nFinished batch decoding (%i steps)[%f s]\n", decode_steps, duration_s);
    printf("Tokens per second: %f\n", tokens_per_s);

    llama_sampler_free(sampler);
    llama_free(context->ctx); // context gets created within the tokenizer
}

int main()
{
    // input configs
    const std::vector<std::string> input_prompts = {"What is quantum computing?",
                                                    "What is kernel fusion?",
                                                    "Nvidia stock is very..."};
    const int max_output_tokens = 512;
    // model configs
    const std::string model_path = "/Users/svc_sps/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf";

    ggml_backend_load_all();
    llama_model_and_vocab model_and_vocab = get_model_and_vocab(model_path);

    bool logging = true;
    if (logging == true)
    {
        auto log_id = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::string log_name = "output-logs_";
        log_name += std::to_string(log_id);
        log_name += ".log";
        log_file = fopen(log_name.c_str(), "w+b");
        fprintf(log_file, "decode tokens,decode duration, tokens/s\n");
    }

    for (int prompt_idx = 0; prompt_idx < input_prompts.size(); ++prompt_idx)
    {
        printf("\nProcessing prompt %i -> %s\n", prompt_idx + 1, input_prompts[prompt_idx].c_str());
        llama_tokenizer tokenizer = tokenize_prompt(&model_and_vocab, input_prompts[prompt_idx], max_output_tokens);
        batch_decode(model_and_vocab.model, model_and_vocab.vocab, &tokenizer.prompt_tokens, &tokenizer.context, tokenizer.n_input_tokens, max_output_tokens, logging);
    }

    llama_model_free(model_and_vocab.model);

    if (logging == true && log_file != NULL)
    {
        fclose(log_file);
    }

    return 0;
}