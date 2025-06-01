import json
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允許跨域請求

class StatsChatbot:
    def __init__(self, knowledge_base_path: str, use_local_llm: bool = False):
        """初始化對話機器人"""
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.use_local_llm = use_local_llm
        
        # Hugging Face API 設定
        self.hf_api_url = "https://api-inference.huggingface.co/models/MediaTek-Research/Llama-Breeze2-3B-Instruct"
        self.hf_api_token = os.getenv('HF_API_TOKEN', '')  # 從環境變數讀取
        
        # 如果要在本地運行模型
        self.tokenizer = None
        self.model = None
        if use_local_llm:
            self._load_local_model()
    
    def _load_local_model(self):
        """載入本地 Breeze2 模型（可選）"""
        try:
            logger.info("正在載入 Llama-Breeze2-3B 模型...")
            model_name = "MediaTek-Research/Llama-Breeze2-3B-Instruct"
            
            # 使用 4-bit 量化以節省記憶體（特別適合 8GB RAM）
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True  # 適合 8GB RAM
            )
            logger.info("模型載入完成")
        except Exception as e:
            logger.error(f"載入本地模型失敗: {e}")
            logger.info("將使用 Hugging Face API")
            self.use_local_llm = False
        
    def load_knowledge_base(self, path: str) -> List[Dict]:
        """載入知識庫"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 如果是陣列格式就直接回傳，如果是物件格式就取questions
                if isinstance(data, list):
                    return data
                else:
                    return data.get('questions', [])
        except Exception as e:
            logger.error(f"載入知識庫失敗: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """文本預處理"""
        # 移除多餘空格
        text = re.sub(r'\s+', ' ', text.strip())
        # 轉換為小寫（保留中文）
        text = text.lower()
        return text
    
    def calculate_similarity(self, query: str, reference: str) -> float:
        """計算兩個字串的相似度（簡易版）"""
        query = self.preprocess_text(query)
        reference = self.preprocess_text(reference)
        
        # 字元級別的 Jaccard 相似度
        query_chars = set(query)
        ref_chars = set(reference)
        
        if not query_chars or not ref_chars:
            return 0.0
            
        intersection = query_chars & ref_chars
        union = query_chars | ref_chars
        
        # 額外考慮關鍵詞匹配
        keywords_similarity = self._keyword_similarity(query, reference)
        
        # 綜合相似度
        jaccard_sim = len(intersection) / len(union)
        return 0.7 * jaccard_sim + 0.3 * keywords_similarity
    
    def _keyword_similarity(self, query: str, reference: str) -> float:
        """關鍵詞相似度計算"""
        # 定義統計系相關關鍵詞（根據你的資料調整）
        keywords = ['統計', '課程', '必修', '選修', '學分', '教授', '考試', 
                   '作業', '成績', '畢業', '研究所', '工作', '實習', '程式',
                   'R語言', 'Python', 'SAS', 'SPSS', '機率', '數理統計',
                   '交換', '國際', '企業', '服務活動', '通識', '擋修',
                   '機率論', '迴歸分析', '線性代數', '微積分']
        
        query_keywords = [kw for kw in keywords if kw in query]
        ref_keywords = [kw for kw in keywords if kw in reference]
        
        if not ref_keywords:
            return 0.0
            
        matched = len(set(query_keywords) & set(ref_keywords))
        return matched / len(ref_keywords)
    
    def search_knowledge_base(self, query: str, threshold: float = 0.5) -> Optional[Dict]:
        """在知識庫中搜尋相關問答"""
        best_match = None
        best_score = 0.0
        
        for qa_pair in self.knowledge_base:
            # 計算與主問題的相似度
            score = self.calculate_similarity(query, qa_pair['question'])
            
            # 檢查變體問題 (variants)
            if 'variants' in qa_pair:
                for variant in qa_pair['variants']:
                    variant_score = self.calculate_similarity(query, variant)
                    score = max(score, variant_score)
            
            # 檢查關鍵詞
            if 'keywords' in qa_pair:
                keywords = qa_pair['keywords'].split(',')
                for keyword in keywords:
                    if keyword.strip() in query:
                        score += 0.15  # 關鍵詞加分
            
            if score > best_score:
                best_score = score
                best_match = qa_pair
        
        logger.info(f"查詢: '{query}' - 最佳匹配分數: {best_score}")
        
        if best_score >= threshold:
            return best_match
        return None
    
    def query_llm(self, query: str, context: str = "") -> str:
        """調用 Breeze-7B 生成回答"""
        # 構建提示詞（Breeze 格式）
        system_prompt = """你是一個專為統計系學生設計的智能助手。
        請用友善、專業的繁體中文回答問題。
        如果問題涉及統計專業知識，請提供準確且易懂的解釋。
        回答要簡潔明瞭，適合大學生理解。"""
        
        if context:
            full_prompt = f"{system_prompt}\n\n參考資訊：{context}\n\n學生問題：{query}\n\n回答："
        else:
            full_prompt = f"{system_prompt}\n\n學生問題：{query}\n\n回答："
        
        # 使用本地模型
        if self.use_local_llm and self.model and self.tokenizer:
            return self._query_local_model(full_prompt)
        
        # 使用 Hugging Face API
        return self._query_hf_api(full_prompt)
    
    def _query_local_model(self, prompt: str) -> str:
        """使用本地 Breeze2 模型生成回答"""
        try:
            # Breeze2 使用標準的 Llama chat template
            messages = [
                {"role": "system", "content": "你是一個專為統計系學生設計的智能助手。請用友善、專業的繁體中文回答問題。"},
                {"role": "user", "content": prompt}
            ]
            
            # 使用 tokenizer 的 chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # 減少 token 數以節省記憶體
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"本地模型生成錯誤: {e}")
            return "抱歉，系統處理您的問題時發生錯誤。"
    
    def _query_hf_api(self, prompt: str) -> str:
        """使用 Hugging Face API 查詢 Breeze2"""
        try:
            headers = {
                "Authorization": f"Bearer {self.hf_api_token}"
            } if self.hf_api_token else {}
            
            # Breeze2 的對話格式
            messages = [
                {"role": "system", "content": "你是一個專為統計系學生設計的智能助手。請用友善、專業的繁體中文回答問題。"},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "inputs": messages,
                "parameters": {
                    "max_new_tokens": 300,  # 適合 3B 模型
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.hf_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '抱歉，我無法生成回答。')
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text']
                return '抱歉，我無法生成回答。'
            else:
                logger.error(f"HF API 錯誤: {response.status_code} - {response.text}")
                if response.status_code == 503:
                    return "AI 模型正在載入中，請稍後再試（約需 20 秒）。"
                return "抱歉，AI 服務暫時無法使用。請稍後再試。"
                
        except requests.exceptions.ConnectionError:
            logger.error("無法連接到 Hugging Face API")
            return "抱歉，無法連接到 AI 服務。請檢查網路連線。"
        except Exception as e:
            logger.error(f"API 查詢錯誤: {e}")
            return "抱歉，系統處理您的問題時發生錯誤。"
    
    def get_response(self, query: str) -> Dict:
        """獲取機器人回應"""
        start_time = datetime.now()
        
        # 首先嘗試從知識庫匹配
        kb_match = self.search_knowledge_base(query)
        
        if kb_match:
            response_time = (datetime.now() - start_time).total_seconds()
            return {
                "answer": kb_match['answer'],
                "source": "knowledge_base",
                "confidence": "high",
                "response_time": response_time,
                "category": kb_match.get('category', '一般問題')
            }
        
        # 如果知識庫沒有匹配，使用 LLM
        logger.info("知識庫無匹配，調用 LLM...")
        
        # 提供一些上下文給 LLM
        context = self._get_relevant_context(query)
        llm_response = self.query_llm(query, context)
        
        response_time = (datetime.now() - start_time).total_seconds()
        return {
            "answer": llm_response,
            "source": "llm",
            "confidence": "medium",
            "response_time": response_time,
            "category": self._categorize_question(query)
        }
    
    def _get_relevant_context(self, query: str) -> str:
        """獲取相關上下文資訊"""
        contexts = []
        
        # 根據關鍵詞提供相關上下文
        if any(word in query for word in ['課程', '必修', '選修', '學分']):
            contexts.append("統計系畢業需修滿128學分，其中必修88學分。包含基本知能、通識課程及專業必修課程。")
        
        if any(word in query for word in ['工作', '就業', '職涯', '實習']):
            contexts.append("統計系畢業生可從事數據分析師、精算師、市場研究員等職位。系上提供企業實習機會，可獲得實習薪資。")
        
        if any(word in query for word in ['程式', 'R', 'Python', '軟體', 'SAS']):
            contexts.append("統計系必修程式課程包括 R語言設計和 SAS程式設計。這些是數據分析的重要工具。")
        
        if any(word in query for word in ['交換', '國際', '出國']):
            contexts.append("統計系提供與廈門大學、西南財經大學等姊妹校的交換計畫，適合大三、大四學生申請。")
        
        if any(word in query for word in ['擋修', '機率論', '數理統計']):
            contexts.append("核心必修科目有擋修規定，機率論和數理統計需依序修習，前一科未達40分不得修習下一科。")
        
        return " ".join(contexts)
    
    def _categorize_question(self, query: str) -> str:
        """對問題進行分類"""
        categories = {
            '課程資訊': ['課程', '必修', '選修', '學分', '上課', '擋修', '通識'],
            '程式技能': ['程式', 'R', 'Python', 'SAS', 'SPSS', '程式語言'],
            '職涯發展': ['工作', '就業', '職涯', '薪水', '產業'],
            '學術研究': ['研究所', '論文', '研究', '教授', '專題'],
            '考試作業': ['考試', '作業', '成績', '評分'],
            '國際交流': ['交換', '國際', '出國', '姊妹校'],
            '實習資訊': ['實習', '企業', '公司']
        }
        
        for category, keywords in categories.items():
            if any(keyword in query for keyword in keywords):
                return category
        
        return '一般問題'

# 初始化聊天機器人
chatbot = None

@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """處理聊天請求"""
    try:
        data = request.get_json()
        user_query = data.get('message', '')
        
        if not user_query:
            return jsonify({
                'error': '請輸入您的問題',
                'status': 'error'
            }), 400
        
        # 獲取機器人回應
        response = chatbot.get_response(user_query)
        
        return jsonify({
            'status': 'success',
            'data': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"聊天處理錯誤: {e}")
        return jsonify({
            'error': '處理您的問題時發生錯誤',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'knowledge_base_loaded': chatbot is not None
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """獲取系統統計資訊"""
    if chatbot:
        categories_count = {}
        for q in chatbot.knowledge_base:
            cat = q.get('category', '一般問題')
            categories_count[cat] = categories_count.get(cat, 0) + 1
        
        return jsonify({
            'total_qa_pairs': len(chatbot.knowledge_base),
            'categories': list(categories_count.keys()),
            'category_counts': categories_count
        })
    return jsonify({'error': '系統尚未初始化'}), 500

if __name__ == '__main__':
    # 初始化聊天機器人
    knowledge_base_path = 'statistic_qa_data.json'  # 使用你的檔案名稱
    
    if not os.path.exists(knowledge_base_path):
        logger.error(f"找不到知識庫檔案: {knowledge_base_path}")
        logger.info("請確保 statistic_qa_data.json 檔案與程式在同一目錄")
        exit(1)
    
    # 檢查是否要使用本地模型
    use_local = os.getenv('USE_LOCAL_MODEL', 'false').lower() == 'true'
    
    chatbot = StatsChatbot(knowledge_base_path, use_local_llm=use_local)
    logger.info(f"成功載入 {len(chatbot.knowledge_base)} 筆問答資料")
    
    if not chatbot.hf_api_token and not use_local:
        logger.warning("未設定 HF_API_TOKEN，將使用免費配額（可能有限制）")
        logger.info("建議設定環境變數: export HF_API_TOKEN='your_token_here'")
    
    # 啟動伺服器
    app.run(debug=True, host='0.0.0.0', port=5001)