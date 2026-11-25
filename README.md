# mba-llm-deobfuscation

Évaluation de LLM pour la désobfuscation d'expressions MBA, en boîte noire. Le modèle reçoit des paires entrée/sortie et propose une expression courte. Z3 vérifie ensuite l'équivalence.

Installation : `pip install -r requirements.txt`, puis copier `.env.example` en `.env` et renseigner la clé du fournisseur choisi. Ollama n'a pas besoin de clé.

Exemple : `python -m experiments.run_experiment third_party/xyntia/datasets/b2 --provider ollama --model codestral:22b --method single --vars 2 --limit 10`

Les résultats sont écrits dans `results/raw/` (un fichier JSONL par run). Les réponses des modèles sont mises en cache dans `results/cache/` : relancer la même expérience n'appelle pas le modèle.

Les jeux de données viennent de Xyntia (`github.com/binsec/xyntia`) et ne sont pas redistribués ici. Xyntia est sous licence LGPL-2.1.