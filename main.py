import sys
import shlex
import cmd
import os
import json
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from rich.logging import RichHandler
from rich.prompt import Prompt
from src.downloader import sync_files
from src.parser import parse_all
from src.ai import DeepSeekAgent
from src.vector_store import VectorStore

# Configure Rich Console and Logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger("PubMed")

# Load environment variables
load_dotenv()

# Initialize VectorStore lazily
vector_store = None

def get_vector_store():
    global vector_store
    if vector_store is None:
        try:
            vector_store = VectorStore()
        except Exception as e:
            logger.error(f"初始化向量数据库失败: {e}")
            return None
    return vector_store

def find_candidates(keyword, limit=20, use_vector=False):
    """
    Search for candidates in metadata.jsonl or via VectorStore.
    Returns a list of dictionaries.
    """
    if use_vector:
        vs = get_vector_store()
        if vs:
            try:
                return vs.search(keyword, limit=limit)
            except Exception as e:
                logger.warning(f"向量搜索失败: {e}。将回退到关键词搜索。")

    # Fallback to keyword search
    metadata_file = os.path.join(os.path.dirname(__file__), "data", "metadata.jsonl")
    matches = []
    
    if not os.path.exists(metadata_file):
        return matches

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                try:
                    data = json.loads(line)
                    text = (data.get("title", "") + " " + data.get("abstract", "")).lower()
                    
                    if keyword.lower() in text:
                        matches.append(data)
                        count += 1
                        if count >= limit:
                            break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"读取元数据时出错: {e}")
        
    return matches

class PubMedShell(cmd.Cmd):
    intro = "" # We will print a custom banner
    prompt = "[bold cyan](PubMed)[/bold cyan] "
    
    def preloop(self):
        banner = """
[bold blue]PubMed 智能文献助手[/bold blue]
[dim]v2.0 - Powered by DeepSeek & ChromaDB[/dim]

输入 [bold green]help[/bold green] 查看使用指南。
输入 [bold green]exit[/bold green] 退出程序。
"""
        console.print(Panel(banner, style="cyan"))

    def do_help(self, arg):
        """显示帮助信息"""
        help_text = """
# 使用指南

欢迎使用 PubMed 智能文献工作台。本软件集成了文献下载、解析、向量检索与 AI 问答功能。

## 常用命令

### 1. sync (同步数据)
从 PubMed FTP 服务器下载最新的 XML 数据文件。
* **用法**: `sync [数量]`
* **示例**: `sync 5` (下载前5个文件)

### 2. parse (解析数据)
将下载的 XML.gz 文件解析为本地可读的 JSONL 格式。
* **用法**: `parse`

### 3. index (构建索引)
将解析后的数据构建为向量索引，以便进行语义检索。建议在每次 parse 后运行一次。
* **用法**: `index [batch_size]`
* **示例**: `index`

### 4. search (搜索文献)
检索本地文献。支持关键词匹配和语义检索。
* **用法**: `search <关键词> [数量] [-v]`
* **参数**:
    - `-v`: 启用语义检索 (需先运行 index)
* **示例**: 
    - `search "lung cancer" 10` (关键词匹配)
    - `search "treatment for headache" -v` (语义检索)

### 5. ask (AI 问答)
利用 DeepSeek AI 回答问题，并基于本地文献提供依据。
* **用法**: `ask <问题>`
* **示例**: `ask "最新肺癌免疫疗法的进展如何？"`

### 6. config (配置)
设置 API Key 等环境变量。
* **用法**: `config <KEY> <VALUE>`
* **示例**: `config DEEPSEEK_API_KEY sk-xxxxx`

### 7. exit
退出程序。
"""
        console.print(Markdown(help_text))

    def do_sync(self, arg):
        limit = None
        if arg:
            try:
                limit = int(arg)
            except ValueError:
                logger.error("参数错误: limit 必须是整数。")
                return
        
        try:
            with console.status("[bold green]正在同步文件...[/bold green]"):
                sync_files(limit=limit)
            console.print("[bold green]同步完成！[/bold green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]操作已取消。[/yellow]")
        except Exception as e:
            logger.exception(f"发生错误: {e}")

    def do_parse(self, arg):
        try:
            # Note: parse_all internally uses tqdm, which might conflict slightly with rich console if not handled carefully,
            # but usually it's fine. We won't wrap it in console.status to let tqdm show progress.
            parse_all()
            console.print("[bold green]解析完成！[/bold green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]操作已取消。[/yellow]")
        except Exception as e:
            logger.exception(f"发生错误: {e}")

    def do_index(self, arg):
        batch_size = 100
        if arg:
            try:
                batch_size = int(arg)
            except ValueError:
                pass
        
        console.print("[cyan]正在初始化向量数据库...[/cyan]")
        vs = get_vector_store()
        if vs:
            metadata_file = os.path.join(os.path.dirname(__file__), "data", "metadata.jsonl")
            try:
                # vs.index_papers uses tqdm, so we don't wrap in status
                vs.index_papers(metadata_file, batch_size=batch_size)
                console.print("[bold green]索引构建完成！[/bold green]")
            except KeyboardInterrupt:
                console.print("\n[yellow]操作已取消。[/yellow]")
            except Exception as e:
                logger.exception(f"索引过程中出错: {e}")
        else:
            logger.error("无法初始化向量数据库，请检查依赖库是否安装。")

    def do_search(self, arg):
        if not arg:
            logger.error("参数错误: 请提供搜索关键字。")
            return
            
        args = shlex.split(arg)
        use_vector = False
        if "-v" in args:
            use_vector = True
            args.remove("-v")
            
        if not args:
            logger.error("请提供搜索关键字。")
            return
            
        keyword = args[0]
        limit = 10
        if len(args) > 1:
            try:
                limit = int(args[1])
            except ValueError:
                logger.error("参数错误: limit 必须是整数。")
                return

        search_type = "语义检索" if use_vector else "关键词匹配"
        console.print(f"正在搜索 [bold cyan]'{keyword}'[/bold cyan] ({search_type})...")
        
        matches = find_candidates(keyword, limit, use_vector=use_vector)

        if not matches:
            console.print("[yellow]未找到匹配项。[/yellow]")
            return

        table = Table(title=f"搜索结果: {keyword} ({len(matches)})")
        table.add_column("PMID", style="cyan", no_wrap=True)
        table.add_column("标题", style="white")
        table.add_column("年份", style="green")
        table.add_column("期刊", style="magenta")

        for m in matches:
            table.add_row(
                m['pmid'],
                m['title'][:100] + "..." if len(m['title']) > 100 else m['title'],
                m.get('year', 'N/A'),
                m.get('journal', 'N/A')
            )
        
        console.print(table)

    def do_config(self, arg):
        args = shlex.split(arg)
        if len(args) != 2:
            logger.error("用法错误。示例: config DEEPSEEK_API_KEY sk-12345")
            return
            
        key, value = args
        env_file = ".env"
        
        # Update .env file
        lines = []
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                lines = f.readlines()
        
        key_found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                key_found = True
            else:
                if not line.endswith("\n"):
                    line += "\n"
                new_lines.append(line)
        
        if not key_found:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            new_lines.append(f"{key}={value}\n")
            
        with open(env_file, "w") as f:
            f.writelines(new_lines)
            
        os.environ[key] = value
        console.print(f"[green]配置已更新: {key} 已保存。[/green]")

    def do_ask(self, arg):
        if not arg:
            logger.error("请提供问题。")
            return
            
        try:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            agent = DeepSeekAgent(api_key=api_key)
        except ValueError as e:
            logger.error(f"无法初始化 AI Agent: {e}")
            console.print("[yellow]请使用 'config DEEPSEEK_API_KEY <your_key>' 设置 API Key。[/yellow]")
            return

        with console.status("[cyan]正在分析问题并提取关键词...[/cyan]"):
            keyword = agent.extract_keywords(arg)
        console.print(f"检索关键词: [bold]{keyword}[/bold]")
        
        candidates = []
        vs = get_vector_store()
        
        if vs:
            console.print("[dim]尝试使用语义检索...[/dim]")
            try:
                candidates = vs.search(arg, limit=10)
                if not candidates:
                    console.print("[dim]语义检索未找到结果，尝试关键词检索...[/dim]")
            except Exception as e:
                logger.warning(f"语义检索出错 ({e})，转为关键词检索...")
        
        if not candidates:
            candidates = find_candidates(keyword, limit=10, use_vector=False)
        
        if not candidates:
            console.print(f"[red]未找到关于 '{keyword}' 的相关文献。尝试换个问法？[/red]")
            return
            
        console.print(f"[green]找到 {len(candidates)} 篇相关文献，正在生成回答...[/green]\n")
        console.rule("[bold blue]AI 回答[/bold blue]")
        
        try:
            response_text = ""
            for chunk in agent.chat(arg, candidates):
                response_text += chunk
                console.print(chunk, end="", highlight=False, markup=False) # Print raw chunk to stream
            console.print("\n")
            console.rule("[bold blue]结束[/bold blue]")
        except KeyboardInterrupt:
            console.print("\n[yellow]回答中止。[/yellow]")
        except Exception as e:
            logger.exception(f"\n发生错误: {e}")

    def do_exit(self, arg):
        """退出程序。"""
        console.print("[bold blue]再见！[/bold blue]")
        return True
    
    def do_quit(self, arg):
        """退出程序。"""
        return self.do_exit(arg)

if __name__ == '__main__':
    try:
        PubMedShell().cmdloop()
    except KeyboardInterrupt:
        print("\n程序已退出。")
