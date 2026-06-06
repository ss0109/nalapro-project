# Text Classification on 20 Newsgroups: Classical Features, Fine-Tuned Transformers, and LLMs

Final project for the **NALAPRO** course (MSc), Lucerne University of Applied
Sciences and Arts (HSLU). The project compares a spectrum of text-classification
methods on the **20 Newsgroups** corpus (20-way topic classification), under a
single preprocessing and evaluation pipeline — from classical bag-of-words and
word-embedding features, through a fine-tuned BERT encoder, to large language
models used zero-shot, few-shot, and with QLoRA parameter-efficient fine-tuning.

## Headline results

**Full 20-class test set (7,532 documents)**

|Method|Accuracy|Macro F1|
|-|:-:|:-:|
|SimpleNN — word2vec (1 epoch)|0.272|0.236|
|SimpleNN — word2vec (30 epochs)|0.642|0.625|
|SimpleNN — TF-IDF|**0.679**|**0.670**|
|SimpleNN — IDF-weighted word2vec|0.634|0.619|
|Fine-tuned BERT (`bert-base-uncased`)|0.690|0.665|
|BERT + domain-adaptive MLM|0.691|0.668|

**Balanced 100-document subset** (used for the LLM comparisons)

|Method|Accuracy|Macro F1|
|-|:-:|:-:|
|Llama-3.1-8B zero-shot|0.620|0.637|
|Llama-3.1-8B few-shot (1/class)|0.440|0.428|
|Fine-tuned BERT (same subset)|**0.710**|**0.690**|
|Llama-3.2-3B (4-bit) zero-shot|0.170|0.165|
|Llama-3.2-3B (4-bit) few-shot|0.000|0.000|
|Llama-3.2-3B (4-bit) QLoRA|0.530|0.502|

> The two groups use \*\*different test sets\*\* and are not directly comparable;
> compare methods within a group, not across groups.

!\[Accuracy of all methods across Tasks 1–5](figures/06\_all\_accuracies.png)

**Takeaways.** For this closed-set task, task-specific fine-tuning of a compact
encoder (BERT) is the strongest approach; a simple TF-IDF vector is a strong and
cheap classical baseline; domain-adaptive MLM gives only a marginal gain on this
small in-domain corpus; in-context learning underperforms fine-tuning and is
sensitive to prompt/parsing design; and QLoRA recovers much of the gap at a tiny
fraction of the cost.

## Tasks

1. **Task 1 — Classical features + shallow NN.** A fixed single-hidden-layer
network trained on four document representations: averaged word2vec (1 and
30 epochs), TF-IDF, and IDF-weighted word2vec.
2. **Task 2 — BERT fine-tuning.** Full fine-tuning of `bert-base-uncased`.
3. **Task 3 — Domain-adaptive MLM.** Continue masked-language-modelling on the
in-domain corpus, then fine-tune for classification.
4. **Task 4 — Llama-3 in-context vs BERT.** Zero-shot and few-shot classification
with a Llama-3 instruction model, compared against fine-tuned BERT on a common
100-document subset.
5. **Task 5 (bonus) — QLoRA.** Parameter-efficient fine-tuning of a 4-bit
Llama-3.2-3B model on the same subset.

!\[Task 5 (bonus): zero-shot vs few-shot vs QLoRA](figures/05\_bonus\_comparison.png)

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example                      # template for secrets (copy to .env)
├── .gitignore
├── notebooks/
│   ├── nlp\_project\_main.ipynb        # Tasks 1–4
│   └── nlp\_project\_bonus\_qlora.ipynb # Task 5 (bonus)
├── results/                          # metrics logged from the runs (JSON)
│   ├── 01\_task1\_results.json
│   ├── 02\_bert\_results.json
│   ├── 03\_task3\_results.json
│   └── 04\_llama\_results.json
├── figures/                          # all figures used in the report
│   ├── 00\_class\_distribution.png
│   ├── 01\_w2v\_epochs.png
│   ├── 01\_task1\_comparison.png
│   ├── 02\_bert\_confusion.png
│   ├── 03\_task2\_vs\_task3.png
│   ├── 04\_llama\_vs\_bert.png
│   ├── 05\_bonus\_comparison.png
│   └── 06\_all\_accuracies.png
└── docs/                             # put your slides + compiled report here
    ├── README.md
    └── report/                       # LaTeX source of the report (IEEE 2-column)
        ├── nalapro\_report\_2col.tex
        ├── README.md
        └── \*.png
```

## Setup

```bash
git clone https://github.com/<your-username>/nalapro-20news-text-classification.git
cd nalapro-20news-text-classification

python -m venv .venv \&\& source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# one-off NLTK data used by the Task 1 preprocessing
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# secrets (see "Credentials" below)
cp .env.example .env   # then edit .env
```

A CUDA GPU is required for the Transformer and LLM tasks; the 4-bit quantization
in Task 4 (fallback model) and the QLoRA bonus need `bitsandbytes` + GPU. The
project was developed on an NVIDIA L4 (Google Colab).

## Credentials

The notebooks **do not contain any API keys**. They read two secrets from the
environment:

* `WANDB\_API\_KEY` — for Weights \& Biases experiment logging
* `HF\_TOKEN` — Hugging Face token (only needed for the *gated* Llama-3.2 model;
the notebooks otherwise fall back to an ungated model)

Set them either by copying `.env.example` to `.env` and filling it in, by
exporting them in your shell, or — in Google Colab — via the Secrets panel
(`from google.colab import userdata`).

## Running

Open the notebooks in Jupyter or Google Colab on a GPU runtime and run top to
bottom. The 20 Newsgroups corpus is downloaded automatically through
scikit-learn's `fetch\_20newsgroups`; the data, pickled datasets, and model
checkpoints are intentionally **not** committed (they are listed in
`.gitignore`).

> \*\*Note on paths:\*\* the notebooks were written for a Colab workflow and use a
> `PROJECT\_ROOT` variable plus an uploaded archive of the prepared data. If you
> run them elsewhere, set `PROJECT\_ROOT` to your working directory and run the
> data-setup cells first to (re)generate the datasets.

## Report and slides

The compiled report PDF and your presentation go in [`docs/`](docs/). The LaTeX
source of the report (IEEE two-column, `IEEEtran`) is in
[`docs/Sparsh Sopory NALAPRO Report.pdf`](docs/Sparsh%20Sopory%20NALAPRO%20Report.pdf) and compiles with pdfLaTeX on Overleaf.

## Tech stack

Python · scikit-learn · gensim · NLTK · PyTorch · Hugging Face Transformers /
PEFT · bitsandbytes · Weights \& Biases.

## Generative AI declaration

Generative AI tools were used as a writing and coding aid (drafting prose,
helping structure the implementation, and debugging). The experimental design,
the choice of methods and hyperparameters, the execution of all training and
evaluation, the numerical results, and their interpretation are the author's own
work.

## License

Released under the [MIT License](LICENSE).

## Author

**Sparsh Sopory** — MSc, HSLU. (Replace this and the copyright in `LICENSE` with
your own details if needed.)

