from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "hukum khamr"
doc_positif = "Rasulullah mengharamkan jual beli khamr (minuman keras)."
doc_negatif = "Nabi shallallahu 'alaihi wasallam makan paha kambing kemudian shalat dan tidak berwudlu lagi."

q_vec = model.encode(query, normalize_embeddings=True)
d_vec_pos = model.encode(doc_positif, normalize_embeddings=True)
d_vec_neg = model.encode(doc_negatif, normalize_embeddings=True)

score_pos = util.cos_sim(q_vec, d_vec_pos)
score_neg = util.cos_sim(q_vec, d_vec_neg)

print(f"[✓] Similarity (positif): {score_pos.item():.4f}")
print(f"[x] Similarity (negatif): {score_neg.item():.4f}")
