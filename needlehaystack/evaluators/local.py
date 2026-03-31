import os

from .evaluator import Evaluator

from langchain_community.chat_models import ChatOpenAI
from langchain.evaluation import load_evaluator


class LocalEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "local-model",
                 base_url: str = None,
                 model_kwargs: dict = None,
                 true_answer: str = None,
                 question_asked: str = None):
        """
        Evaluator using a local model with OpenAI-compatible API.
        
        :param model_name: The name of the local model.
        :param base_url: The base URL for the local model API endpoint.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        # Get base URL from parameter or environment
        self.base_url = base_url or os.getenv('NIAH_EVALUATOR_BASE_URL') or os.getenv('NIAH_MODEL_BASE_URL')
        if not self.base_url:
            raise ValueError("base_url must be provided or NIAH_EVALUATOR_BASE_URL/NIAH_MODEL_BASE_URL must be in env. "
                           "Example: http://localhost:8000/v1")

        self.model_name = model_name
        self.model_kwargs = model_kwargs or self.DEFAULT_MODEL_KWARGS
        self.true_answer = true_answer
        self.question_asked = question_asked

        # API key is optional for local models
        api_key = os.getenv('NIAH_EVALUATOR_API_KEY', 'dummy-key')
        self.api_key = api_key
        
        # Use ChatOpenAI with custom base_url for local model
        self.evaluator = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            **self.model_kwargs
        )

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])
