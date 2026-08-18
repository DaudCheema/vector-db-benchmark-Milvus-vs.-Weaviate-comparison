import os
import shutil
import torch
import weaviate
import weaviate.classes.config as wvc
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pymilvus import MilvusClient, DataType as MilvusDataType
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

app = FastAPI(title="Milvus vs Weaviate Vector Benchmark Gateway")

milvus_client = None
weaviate_client = None  # One single persistent global client
processor = None
model = None

# Pure local disk path for Milvus Lite
MILVUS_DB_PATH = "./local_milvus_comparison.db"

# ==========================================
# YOUR SECURE WEAVIATE CLOUD CREDENTIALS
# ==========================================
CLUSTER_URL = "https://db-comparsion-iytp1wom.weaviate.network"
WEAVIATE_API_KEY = "Add_your_weaviate_api_key_here"

@app.on_event("startup")
def startup_event():
    global milvus_client, weaviate_client, processor, model
    print("⏳ Loading DinoV2 Model components...")
    
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    
    # 1. Initialize Local Milvus Lite Database
    print("💾 Booting Milvus Lite...")
    milvus_client = MilvusClient(MILVUS_DB_PATH)
    if not milvus_client.has_collection("scenery_benchmark"):
        schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="image_id", datatype=MilvusDataType.VARCHAR, max_length=150, is_primary=True)
        schema.add_field(field_name="vector", datatype=MilvusDataType.FLOAT_VECTOR, dim=768)
        
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection(collection_name="scenery_benchmark", schema=schema, index_params=index_params)
        
    # 2. Establish single global Cloud Pipeline (Bypasses Windows Startup blocks)
    print("🚀 Connecting persistently to Weaviate Cloud...")
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=CLUSTER_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        skip_init_checks=True  # Guarantees Uvicorn boots instantly without blocking
    )
    print("✅ Gateway server initialization complete!")

@app.on_event("shutdown")
def shutdown_event():
    global weaviate_client
    if weaviate_client is not None:
        weaviate_client.close()
        print("🛑 Closed cloud database pipelines cleanly.")

def extract_embedding(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output[0].cpu().numpy().tolist()
    return embedding

@app.post("/insert")
async def insert_vector(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        raw_vector = extract_embedding(temp_path)
        image_name = file.filename
        
        # 1. Store to Local Milvus
        milvus_client.insert(collection_name="scenery_benchmark", data=[{"image_id": image_name, "vector": raw_vector}])
        
        # 2. Store to Persistent Weaviate Cloud Channel
        # Ensure the connection is open lazily right before the call
        if not weaviate_client.is_connected():
            weaviate_client.connect()
            
        if not weaviate_client.collections.exists("SceneryBenchmark"):
            weaviate_client.collections.create(
                name="SceneryBenchmark",
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                vector_index_config=wvc.Configure.VectorIndex.hnsw(distance_metric=wvc.VectorDistances.COSINE),
                properties=[wvc.Property(name="image_id", data_type=wvc.DataType.TEXT)]
            )
            
        weaviate_coll = weaviate_client.collections.get("SceneryBenchmark")
        weaviate_coll.data.insert(properties={"image_id": image_name}, vector=raw_vector)
        
        return {"status": "success", "filename": image_name, "vector_dim": len(raw_vector)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search")
async def search_vector(file: UploadFile = File(...), limit: int = Query(default=3)):
    temp_path = f"temp_query_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        query_vector = extract_embedding(temp_path)
        
        # 1. Query Milvus Lite
        milvus_client.load_collection("scenery_benchmark")
        milvus_res = milvus_client.search(
            collection_name="scenery_benchmark", data=[query_vector], limit=limit, output_fields=["image_id"]
        )
        milvus_parsed = [{"image_id": hit["entity"]["image_id"], "score": hit["distance"]} for hit in milvus_res[0]]
        
        # 2. Query Persistent Weaviate Cloud Channel
        if not weaviate_client.is_connected():
            weaviate_client.connect()
            
        weaviate_parsed = []
        if weaviate_client.collections.exists("SceneryBenchmark"):
            weaviate_coll = weaviate_client.collections.get("SceneryBenchmark")
            weaviate_res = weaviate_coll.query.near_vector(
                near_vector=query_vector, limit=limit, return_properties=["image_id"]
            )
            weaviate_parsed = [{"image_id": obj.properties["image_id"], "score": obj.metadata.distance} for obj in weaviate_res.objects]
        
        return {"query_file": file.filename, "milvus_results": milvus_parsed, "weaviate_results": weaviate_parsed}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)