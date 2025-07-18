from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="WesPro/Mistral-Small-3.1-24B-Instruct-2503-HF-Q6_K-GGUF",
	filename="mistral-small-3.1-24b-instruct-2503-hf-q6_k.gguf",
)

output = llm(
	"Once upon a time,",
	max_tokens=512,
	echo=True
)
print(output)