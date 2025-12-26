from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import logging , asyncio , os 

current_dir = Path(__file__).parent
env_file = current_dir / 'connection.env'
load_dotenv(env_file)

class MongoDB:
    def __init__(self , uri : str , db_name : str , collection_name : str):
        self.uri = uri 
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None 
        self.db = None 
        self.collection = None

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(self.uri )
            self.db = self.client[self.db_name] 
            self.collection = self.db[self.collection_name]

            await self.client.admin.command('ping')
            logging.info('Successfully')

        except Exception as e : 
            logging.error(f'Error {e}')
            raise ConnectionFailure('Not respond')
    
    async def close(self):
        if self.client:
            self.client.close()
            logging.info('Disconnect !!!')
    
    async def save_predict(self, filename: str, label: str, confidence: str, probs: dict, gradcam_base64: str):
        if self.collection is None:
            await self.connect()
        
        document = {
            "filename": filename,
            "label": label.upper(),
            "confidence": confidence,
            "probabilities": probs,
            "gradcam_img": gradcam_base64,
            "created_at": datetime.utcnow(),
            "status": "completed"
        }

        result = await self.collection.insert_one(document)
        return str(result.inserted_id)
    

async def run():
    db = MongoDB(uri=os.getenv('HOST_NSQL') , db_name=os.getenv('DB_NAME') , collection_name=os.getenv('COL_NAME'))
    try:
        print("--- Đang kết nối ---")
        await db.connect()
        print("--- Đang lưu thử dữ liệu ---") 
        res_id = await db.save_predict(
            filename="test_image.jpg",
            label="NORMAL",
            confidence="99%",
            probs={"NORMAL": 0.99, "PNEUMONIA": 0.01},
            gradcam_base64="base64_string_here"
        )
        print(f"Thành công! ID: {res_id}")
    except Exception as e:
        print(f"Lỗi khi test: {e}")
    finally:
        await db.close()
   

if __name__ == "__main__":
    asyncio.run(run())

