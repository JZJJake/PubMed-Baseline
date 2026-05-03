import asyncio
import os
import sqlite3
from scraper import crawl_worker
import db_manager

async def test_run():
    for f in ['scraper.db', 'scraper_state.db']:
        if os.path.exists(f): os.remove(f)
    task_id = "test-task"
    url = "https://m.x-mol.com/paper/tag/academicArea/chem"
    db_manager.init_db()
    db_manager.create_task(task_id, url, url)
    print("Testing spider manager logic...")
    task = asyncio.create_task(crawl_worker(task_id, url, headless=True))
    await asyncio.sleep(5)
    task.cancel()
    print("Spider test completed.")

if __name__ == "__main__":
    asyncio.run(test_run())
