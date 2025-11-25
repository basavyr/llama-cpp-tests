#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct ModelParams
{
    int n_gpu_layers;
    std::string model_path;
    llama_model_params model_params;
};

int main()
{
    std::string title = "Testing inference using `llama.cpp`";
    printf("%s\n", title.c_str());

    ggml_backend_load_all();
    const std::string MODEL_PATH = "/Users/svc_sps/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf";

    ModelParams model_config = ModelParams();
    model_config.n_gpu_layers = 99;
    model_config.model_path = MODEL_PATH;
    model_config.model_params = llama_model_default_params();

    llama_model *model = llama_model_load_from_file(model_config.model_path.c_str(), model_config.model_params);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    return 0;
}