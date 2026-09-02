from data_loader import data_loader
import embedding
pipeline = embedding.EmbeddingPipeline()
docs=data_loader();

chunks=pipeline.chunk_documents(docs)
vectors=pipeline.embed_chunks(chunks)