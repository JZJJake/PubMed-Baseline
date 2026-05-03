from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import os
import webbrowser
import asyncio
import sys

import db_manager
from scraper import crawl_worker, task_events
import pubmed_src.downloader as pubmed_downloader
import pubmed_src.parser as pubmed_parser
import pubmed_src.vector_store as pubmed_vs

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

static_dir = os.path.join(application_path, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

from contextlib import asynccontextmanager

def install_playwright_browsers():
    print("Checking/installing Playwright browsers...")
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    if getattr(sys, 'frozen', False):
        print("Running as a frozen executable. Skipping automatic playwright install.")
        return
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        print("Playwright browsers ready.")
    except Exception as e:
        print(f"Warning: Failed to install playwright browsers automatically. Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    install_playwright_browsers()
    async def open_browser_async():
        await asyncio.sleep(0.5)
        webbrowser.open('http://127.0.0.1:8000')
    asyncio.create_task(open_browser_async())
    yield
    pass

app = FastAPI(title="Web Scraper Client", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class ScrapeRequest(BaseModel):
    url: str
    show_browser: bool = True
    update_data: bool = False

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/console", response_class=HTMLResponse)
async def get_console():
    console_path = os.path.join(static_dir, "console.html")
    with open(console_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/scrape/start")
async def start_scraping(request: ScrapeRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, request.url))
    if request.update_data:
        await asyncio.to_thread(db_manager.clear_task_data, task_id)
    task = await asyncio.to_thread(db_manager.get_task, task_id)
    if not task:
        await asyncio.to_thread(db_manager.create_task, task_id, request.url, request.url)
    if task_id in task_events and not task_events[task_id]['stop'].is_set():
         return {"task_id": task_id, "status": "already running or paused"}
    await asyncio.to_thread(db_manager.update_task_status, task_id, "running")
    background_tasks.add_task(crawl_worker, task_id, request.url, not request.show_browser)
    return {"task_id": task_id, "status": "started"}

@app.get("/api/scrape/status/{task_id}")
async def get_scraping_status(task_id: str):
    task = await asyncio.to_thread(db_manager.get_task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    active_count = await asyncio.to_thread(db_manager.get_active_count, task_id)
    return {
        "status": task['status'],
        "pages_scraped": task['total_scraped'],
        "current_url": f"{active_count} 个页面正在队列中...",
        "is_running": task['status'] == 'running'
    }

@app.get("/api/scrape/tree/{task_id}")
async def get_scrape_tree(task_id: str):
    tree_data = await asyncio.to_thread(db_manager.get_url_tree, task_id)
    return {"tree": tree_data}

@app.post("/api/scrape/pause/{task_id}")
async def pause_scraping(task_id: str):
    if task_id in task_events:
        task_events[task_id]['pause'].clear()
        await asyncio.to_thread(db_manager.update_task_status, task_id, "paused")
    return {"status": "paused"}

@app.post("/api/scrape/resume/{task_id}")
async def resume_scraping(task_id: str):
    if task_id in task_events:
        task_events[task_id]['pause'].set()
        await asyncio.to_thread(db_manager.update_task_status, task_id, "running")
    return {"status": "resumed"}

@app.post("/api/scrape/stop/{task_id}")
async def stop_scraping(task_id: str):
    if task_id in task_events:
        task_events[task_id]['stop'].set()
        task_events[task_id]['pause'].set()
    await asyncio.to_thread(db_manager.update_task_status, task_id, "stopped")
    return {"status": "stopped"}

@app.post("/api/pubmed/sync")
async def pubmed_sync(background_tasks: BackgroundTasks):
    def run_sync():
        print("Starting PubMed sync...")
        pubmed_downloader.sync_files()
        print("Starting PubMed parse...")
        pubmed_parser.parse_all()
        print("PubMed pipeline finished!")
    background_tasks.add_task(run_sync)
    return {"status": "PubMed sync started"}

@app.post("/api/pubmed/index")
async def pubmed_index(background_tasks: BackgroundTasks):
    def run_index():
        print("Starting PubMed vector index...")
        vs = pubmed_vs.get_vector_store()
        metadata_file = os.path.join(os.path.dirname(__file__), "data", "metadata.jsonl")
        vs.index_papers(metadata_file, batch_size=100)
        print("PubMed index finished!")
    background_tasks.add_task(run_index)
    return {"status": "PubMed indexing started"}

if __name__ == "__main__":
    import uvicorn
    # Start FastAPI server directly as it did before
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
