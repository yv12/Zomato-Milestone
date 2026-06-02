from app.data.repository import RestaurantRepository
from app.services.filter_service import FilterService
from app.services.vector_search import VectorSearchEngine
from app.services.llm_client import LLMClient
from app.services.orchestrator import RecommendationOrchestrator

class ServiceRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized and self.repo is not None:
            return
        
        print("Initializing central service registry...")
        try:
            self.repo = RestaurantRepository()
            self.filter_service = FilterService(self.repo)
            self.vector_search_engine = VectorSearchEngine(self.repo)
            self.llm_client = LLMClient()
            self.orchestrator = RecommendationOrchestrator(
                self.filter_service, 
                self.vector_search_engine, 
                self.llm_client
            )
            print("Successfully initialized all Zomato discovery services in registry.")
        except Exception as e:
            print(f"Error initializing services in registry: {e}")
            self.repo = None
            self.filter_service = None
            self.vector_search_engine = None
            self.llm_client = None
            self.orchestrator = None
            
        self._initialized = True

services = ServiceRegistry()
