
import os
import json
import gzip
from lxml import etree
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
METADATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "metadata.jsonl")

def extract_text(element):
    """辅助函数：从 XML 元素提取文本，处理 None 情况。"""
    return element.text if element is not None and element.text else ""

def parse_article(elem):
    """
    解析单个 PubmedArticle 元素并返回字典。
    """
    try:
        medline = elem.find("MedlineCitation")
        pubmed_data = elem.find("PubmedData")
        
        if medline is None:
            return None

        # PMID
        pmid_elem = medline.find("PMID")
        pmid = extract_text(pmid_elem)

        article = medline.find("Article")
        if article is None:
            return None

        # 标题
        title_elem = article.find("ArticleTitle")
        title = extract_text(title_elem)

        # 摘要
        abstract_text = ""
        abstract = article.find("Abstract")
        if abstract is not None:
            texts = abstract.findall("AbstractText")
            abstract_text = " ".join([extract_text(t) for t in texts if t.text])

        # 作者
        authors = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = extract_text(author.find("LastName"))
                fore_name = extract_text(author.find("ForeName"))
                if last_name or fore_name:
                    authors.append(f"{last_name} {fore_name}".strip())

        # 期刊
        journal_elem = article.find("Journal")
        journal_title = ""
        year = ""
        if journal_elem is not None:
            journal_title = extract_text(journal_elem.find("Title"))
            journal_issue = journal_elem.find("JournalIssue")
            if journal_issue is not None:
                pub_date = journal_issue.find("PubDate")
                if pub_date is not None:
                    year_elem = pub_date.find("Year")
                    if year_elem is not None:
                        year = extract_text(year_elem)
                    else:
                        # 如果没有 Year，尝试用 MedlineDate
                        medline_date = pub_date.find("MedlineDate")
                        if medline_date is not None:
                            # 通常格式 "1975 May-Jun"
                            year = extract_text(medline_date)[:4]

        # 出版类型
        pub_types = []
        pub_type_list = article.find("PublicationTypeList")
        if pub_type_list is not None:
            for pt in pub_type_list.findall("PublicationType"):
                pub_types.append(extract_text(pt))

        # 语言
        languages = []
        for lang in article.findall("Language"):
            languages.append(extract_text(lang))

        # MeSH 主题词
        mesh_terms = []
        mesh_list = medline.find("MeshHeadingList")
        if mesh_list is not None:
            for mesh in mesh_list.findall("MeshHeading"):
                descriptor = extract_text(mesh.find("DescriptorName"))
                qualifiers = []
                for q in mesh.findall("QualifierName"):
                    qualifiers.append(extract_text(q))
                
                term = descriptor
                if qualifiers:
                    term += f" [{', '.join(qualifiers)}]"
                mesh_terms.append(term)
        
        # 化学物质
        chemicals = []
        chemical_list = medline.find("ChemicalList")
        if chemical_list is not None:
            for chem in chemical_list.findall("Chemical"):
                chemicals.append(extract_text(chem.find("NameOfSubstance")))

        # 关键词
        keywords = []
        keyword_list = medline.find("KeywordList")
        if keyword_list is not None:
            for kw in keyword_list.findall("Keyword"):
                keywords.append(extract_text(kw))

        # 标识符 (DOI, PMCID)
        doi = ""
        pmcid = ""
        if pubmed_data is not None:
            article_id_list = pubmed_data.find("ArticleIdList")
            if article_id_list is not None:
                for article_id in article_id_list.findall("ArticleId"):
                    id_type = article_id.get("IdType")
                    if id_type == "doi":
                        doi = extract_text(article_id)
                    elif id_type == "pmc":
                        pmcid = extract_text(article_id)

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "authors": authors,
            "journal": journal_title,
            "year": year,
            "doi": doi,
            "pmcid": pmcid,
            "pub_types": pub_types,
            "languages": languages,
            "mesh_terms": mesh_terms,
            "chemicals": chemicals,
            "keywords": keywords
        }

    except Exception as e:
        # print(f"Error parsing article {pmid}: {e}")
        return None

def process_file(filepath, output_file):
    """
    解析单个 .xml.gz 文件并将记录追加到输出文件。
    """
    print(f"正在解析 {filepath}...")
    count = 0
    try:
        with gzip.open(filepath, "rb") as f:
            context = etree.iterparse(f, events=("end",), tag="PubmedArticle")
            
            with open(output_file, "a", encoding="utf-8") as out_f:
                for event, elem in context:
                    data = parse_article(elem)
                    if data:
                        json.dump(data, out_f, ensure_ascii=False)
                        out_f.write("\n")
                        count += 1
                    
                    # 清除元素以节省内存
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
    except Exception as e:
        print(f"处理文件 {filepath} 出错: {e}")
        
    print(f"从 {os.path.basename(filepath)} 中提取了 {count} 篇文章。")

def parse_all(raw_dir=RAW_DIR, output_file=METADATA_FILE):
    """
    迭代 raw_dir 中的所有 .xml.gz 文件并进行解析。
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 清除现有文件还是追加？
    # 在此实现中，我们清除它以避免多次运行时重复
    if os.path.exists(output_file):
        os.remove(output_file)

    files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".xml.gz")])
    
    if not files:
        print("未找到可解析的 .xml.gz 文件。")
        return

    for filename in tqdm(files, desc="处理文件"):
        filepath = os.path.join(raw_dir, filename)
        process_file(filepath, output_file)
