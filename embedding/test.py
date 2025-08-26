from graphviz import Digraph
import os

# Buat diagram flowchart baru
dot = Digraph(comment="Alur Ekstraksi & Pengelompokan Kata Kunci (Versi Saran)")
dot.attr(rankdir="TB", size="8,10")
dot.attr("node", shape="rect", style="rounded,filled", color="lightblue", fontname="Helvetica")

# Node utama
dot.node("A", "Query Pengguna")
dot.node("B", "Normalisasi Teks\n(lowercase, hapus stopword,\nlematisasi ringan)")
dot.node("C", "Ekstraksi Kandidat Keyword")
dot.node("C1", "Statistik (TF-IDF / YAKE)")
dot.node("C2", "Embedding Similarity\n(dengan korpus hadis)")
dot.node("C3", "Rule-based\n(Kamus istilah Islam)")
dot.node("D", "Deteksi Frasa Multi-token\n(n-gram: 2–3 kata)")
dot.node("E", "Synonym Expansion\n(‘haram’ → ‘mengharamkan’, dll)")
dot.node("F", "Clustering Embedding\n(K-means / HDBSCAN)")
dot.node("G", "Keyword Map Final")
dot.node("H", "Integrasi ke Retrieval\n(Filter + FAISS Search)")

# Hubungan antar node
dot.edges(["AB"])
dot.edge("B", "C")
dot.edge("C", "C1")
dot.edge("C", "C2")
dot.edge("C", "C3")
dot.edge("C1", "D")
dot.edge("C2", "D")
dot.edge("C3", "D")
dot.edge("D", "E")
dot.edge("E", "F")
dot.edge("F", "G")
dot.edge("G", "H")

# Tampilkan diagram
# Simpan ke direktori yang dapat ditulis, misal: './output'
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "keyword_extraction_flowchart")
dot.render(output_path, format="png", cleanup=False)
print(f"Flowchart saved to: {output_path}.png")
