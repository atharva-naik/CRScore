python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/Comment_Generation/codepatch_data_for_index \
  --index data/Comment_Generation/codepatch_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions