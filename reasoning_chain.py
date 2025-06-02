from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class MultiHopReasoningChain:
    """Çoklu adımlı akıl yürütme zinciri"""
    
    def __init__(self, model: ChatOpenAI = None, retriever = None):
        self.model = model or ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
        self.retriever = retriever
        self.output_parser = StrOutputParser()
        
        # Ana akıl yürütme zinciri için prompt
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen Netpak firmasının gelişmiş bir akıl yürütme asistanısın. 
            Görevin, Netpak'ın politika dokümanları ve verilerini kullanarak net ve kısa cevaplar vermektir.
            
            ÖNEMLİ KURALLAR:
            1. SADECE Netpak'ın verdiği dokümanlar ve veriler üzerinden yanıt ver
            2. Başka kaynaklardan veya genel bilgilerden yanıt verme
            3. Eğer Netpak'ın verilerinde yeterli bilgi yoksa, bunu açıkça belirt
            4. Genel bilgileri ve mantıksal çıkarımları kullan:
               - Eğer bir ürün türü (örn. süt) alerjen olarak belirtilmişse, benzer ürünler (örn. soya sütü) de alerjen olarak değerlendirilmelidir
               - Eğer bir kategori (örn. süt ürünleri) alerjen olarak belirtilmişse, o kategorideki tüm ürünler alerjen olarak değerlendirilmelidir
               - Benzer özellikteki ürünler için aynı kurallar geçerlidir
            
            CEVAP FORMATI:
            1. Kısa ve net bir cevap ver (1-2 cümle)
            2. Eğer bilgi yoksa, "Netpak'ın dokümanlarında bu konuda spesifik bilgi bulunmamaktadır" de
            3. Gereksiz detaylardan kaçın
            4. Tekrarlardan kaçın
            
            Her adımı açıkça belirt ve mantıksal akışı koru."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        logging.info("Initialized reasoning prompt:")
        for msg in self.reasoning_prompt.messages:
            if hasattr(msg, 'prompt'):
                logging.info(f"Role: {msg.prompt.template}")
            else:
                logging.info(f"Role: {msg}")
        
        # Alt soru oluşturma zinciri için prompt
        self.subquestion_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ana soruyu analiz et ve Netpak'ın dokümanlarından cevaplanabilecek alt soruları belirle.
            Her alt soru, Netpak'ın politikaları veya verileri ile ilgili olmalı ve birbirleriyle mantıksal olarak bağlantılı olmalıdır.
            
            ÖNEMLİ: 
            1. Sadece Netpak'ın dokümanlarında bulunabilecek bilgileri sor
            2. Genel bilgileri ve mantıksal çıkarımları da dikkate al
            3. Benzer ürünler ve kategoriler için karşılaştırmalı sorular oluştur
            4. Soruları kısa ve net tut
            
            Alt soruları şu formatta döndür:
            SUBQUESTION_1: [Netpak politikası/verisi ile ilgili alt soru 1]
            SUBQUESTION_2: [Netpak politikası/verisi ile ilgili alt soru 2]
            ...
            SUBQUESTION_N: [Netpak politikası/verisi ile ilgili alt soru n]"""),
            ("human", "{question}")
        ])
        
        logging.info("Initialized subquestion prompt:")
        for msg in self.subquestion_prompt.messages:
            if hasattr(msg, 'prompt'):
                logging.info(f"Role: {msg.prompt.template}")
            else:
                logging.info(f"Role: {msg}")
        
        # Bilgi birleştirme zinciri için prompt
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Netpak'ın dokümanlarından elde edilen bilgileri ve mantıksal çıkarımları kullanarak net ve kısa bir cevap oluştur.
            
            ÖNEMLİ KURALLAR:
            1. SADECE Netpak'ın dokümanlarından gelen bilgileri kullan
            2. Başka kaynaklardan bilgi ekleme
            3. Eksik bilgileri varsayma, bunun yerine bilgi eksikliğini belirt
            4. Mantıksal çıkarımları kullan:
               - Benzer ürünler için aynı kuralları uygula
               - Kategoriler arası ilişkileri değerlendir
               - Genel bilgileri ve çıkarımları kullan
            
            CEVAP FORMATI:
            1. Kısa ve net bir cevap ver (1-2 cümle)
            2. Eğer bilgi yoksa, "Netpak'ın dokümanlarında bu konuda spesifik bilgi bulunmamaktadır" de
            3. Gereksiz detaylardan kaçın
            4. Tekrarlardan kaçın"""),
            ("human", """Netpak Dokümanlarından Bağlam: {context}
            
            Alt Sorular ve Netpak Dokümanlarından Cevapları:
            {subquestion_answers}
            
            Nihai cevabı oluştur.""")
        ])
        
        logging.info("Initialized synthesis prompt:")
        for msg in self.synthesis_prompt.messages:
            if hasattr(msg, 'prompt'):
                logging.info(f"Role: {msg.prompt.template}")
            else:
                logging.info(f"Role: {msg}")
    
    def invoke(self, question: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Soruya yanıt ver"""
        try:
            logging.info(f"\n{'='*50}\nProcessing new question: {question}\n{'='*50}")
            
            # Bağlamı al
            context = ""
            if self.retriever:
                docs = self.retriever.invoke(question)
                context = "\n".join(doc.page_content for doc in docs)
                logging.info(f"Retrieved context length: {len(context)} characters")
                logging.info(f"Context preview: {context[:500]}...")
                logging.info(f"Full context being sent to model:\n{context}")
            
            # Alt soruları oluştur
            logging.info(f"\nCreating subquestions for: {question}")
            subquestions = self._create_subquestions(question)
            logging.info(f"Created subquestions: {subquestions}")
            
            # Her alt soruyu cevapla
            subquestion_answers = []
            for subq in subquestions:
                logging.info(f"\nAnswering subquestion: {subq}")
                answer = self._answer_subquestion(subq, context)
                logging.info(f"Answer for {subq}: {answer[:200]}...")
                subquestion_answers.append({
                    "question": subq,
                    "answer": answer
                })
            
            # Cevapları birleştir
            logging.info("\nSynthesizing final answer...")
            final_answer = self._synthesize_answers(subquestion_answers, context)
            logging.info(f"Final answer: {final_answer[:200]}...")
            
            return final_answer
            
        except Exception as e:
            logging.error(f"Error in multi-hop reasoning process: {str(e)}")
            return f"Error in reasoning process: {str(e)}"
    
    def _create_subquestions(self, question: str) -> List[str]:
        """Ana sorudan alt sorular oluştur"""
        try:
            logging.info(f"Creating subquestions for question: {question}")
            subquestion_chain = self.subquestion_prompt | self.model | self.output_parser
            logging.info(f"Subquestion prompt template:\n{self.subquestion_prompt.messages}")
            result = subquestion_chain.invoke({"question": question})
            logging.info(f"Raw subquestion result: {result}")
            
            # Alt soruları ayır
            subquestions = []
            for line in result.split("\n"):
                if line.startswith("SUBQUESTION_"):
                    subquestion = line.split(":", 1)[1].strip()
                    subquestions.append(subquestion)
            
            logging.info(f"Extracted subquestions: {subquestions}")
            return subquestions
        except Exception as e:
            logging.error(f"Error creating subquestions: {str(e)}")
            return [question]  # Hata durumunda ana soruyu döndür
    
    def _answer_subquestion(self, subquestion: str, context: str) -> str:
        """Alt soruyu cevapla"""
        try:
            logging.info(f"Answering subquestion: {subquestion}")
            logging.info(f"Context length: {len(context)} characters")
            logging.info(f"Full context being sent to model:\n{context}")
            answer_chain = self.reasoning_prompt | self.model | self.output_parser
            logging.info(f"Reasoning prompt template:\n{self.reasoning_prompt.messages}")
            return answer_chain.invoke({
                "question": subquestion,
                "chat_history": [],
                "context": context
            })
        except Exception as e:
            logging.error(f"Error answering subquestion: {str(e)}")
            return f"Error answering: {subquestion}"
    
    def _synthesize_answers(self, subquestion_answers: List[Dict[str, str]], context: str) -> str:
        """Alt soruların cevaplarını birleştir"""
        try:
            # Alt soru cevaplarını formatla
            formatted_answers = "\n".join([
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in subquestion_answers
            ])
            logging.info(f"Formatted answers length: {len(formatted_answers)} characters")
            logging.info(f"Formatted answers preview: {formatted_answers[:500]}...")  # İlk 500 karakteri göster
            logging.info(f"Full formatted answers being sent to model:\n{formatted_answers}")
            logging.info(f"Full context being sent to model:\n{context}")
            
            synthesis_chain = self.synthesis_prompt | self.model | self.output_parser
            logging.info(f"Synthesis prompt template:\n{self.synthesis_prompt.messages}")
            return synthesis_chain.invoke({
                "context": context,
                "subquestion_answers": formatted_answers
            })
        except Exception as e:
            logging.error(f"Error synthesizing answers: {str(e)}")
            return "Error synthesizing final answer" 