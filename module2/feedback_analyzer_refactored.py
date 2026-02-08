"""
COMPONENT-BASED ARCHITECTURE: FLEXIBILITY AND EXTENSIBILITY

Why Component-Based Architecture is More Flexible and Extensible
=================================================================

Separation of Concerns
-----------------------
Each component handles one specific responsibility. `PromptProvider` manages
prompts, `CustomerFeedbackOutput` handles output parsing, and
`CustomerFeedbackAnalyzer` orchestrates the analysis. This makes the code
easier to understand, test, and debug.

Easy to Swap Implementations
-----------------------------
You can replace any component without touching the others. Need a different
output format? Create a new output parser class. Want to try different prompts?
Add a new prompt version or even a completely different prompt provider. Want
to switch LLM providers? Modify only the analyzer.

Independent Testing
-------------------
Each component can be unit tested in isolation. You can test prompt templates
without calling an LLM, validate output parsing with mock data, and verify the
analyzer logic with stub dependencies.

Reusability
-----------
Components can be reused across different parts of your application. The same
`PromptProvider` could serve multiple analyzers, and the same output parser
could work with different prompt templates.

Open for Extension, Closed for Modification
--------------------------------------------
Following the Open/Closed Principle, you can add new functionality (new prompt
versions, new output formats, new analysis types) by creating new classes
rather than modifying existing code, reducing the risk of breaking working
features.

Dependency Injection
--------------------
By passing components as parameters (dependency injection), you make the system
configurable at runtime and easier to mock for testing, while also making
dependencies explicit and visible in the code.
"""
import logging

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# loads OPENAI_API_KEY from .env
load_dotenv()


class CustomerFeedback(BaseModel):
    sentiment: int = Field(description="A sentiment score from 1-5 where 1=very negative, 5=very positive")
    category: str = Field(description="A category: Product, Service, or Billing")


class CustomerFeedbackOutput:
    def __init__(self, response_structure):
        self.output_parser = JsonOutputParser(pydantic_object=response_structure)

    def get_output_instructions(self):
        return self.output_parser.get_format_instructions()

    def get_output_parser(self):
        return self.output_parser


class PromptProvider:
    def __init__(self, version):
        self.version = version

    def get_prompt(self):
        if self.version == 1:
            return \
                """
                You are an expert in customer's feedback analysis. Review the customer feedback and do the following tasks:
                1. Rate the sentiment of this customer feedback on a scale of 1-5 where 1=very negative, 5=very positive. Just respond with the number.
                2. Categorize this customer feedback into one of these categories: Product, Service, or Billing. Respond with just the category name.
                Customer feedback: {feedback_text}
                Format instructions: {format_instructions}
                """

        raise ValueError(f"Invalid version: {self.version}")


class CustomerFeedbackAnalyzer:
    def __init__(self,
                 output: CustomerFeedbackOutput,
                 prompt_provider: PromptProvider,
                 llm_model):
        self.llm_model = llm_model
        self.output = output
        self.prompt_provider = prompt_provider

    def analyze_customer_feedback(self, feedback_text):
        """Analyze customer feedback and return sentiment + category"""

        prompt_template = PromptTemplate(
            template=self.prompt_provider.get_prompt(),
            input_variables=["feedback_text"],
            partial_variables={"format_instructions": self.output.get_output_instructions()}
        )

        chain = prompt_template | self.llm_model | self.output.get_output_parser()

        try:
            result = chain.invoke({"feedback_text": feedback_text})
            return result
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error analyzing feedback: {str(e)}")
            return {
                "sentiment": 0,
                "category": "Unknown"
            }


if __name__ == "__main__":
    prompt_Provider = PromptProvider(version=1)
    output = CustomerFeedbackOutput(CustomerFeedback)

    # set explicitly to show how it might be done
    model = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
    feedback_analyzer = CustomerFeedbackAnalyzer(output, prompt_Provider, model)
    test_feedback = "The new dashboard is confusing and slow to load"

    logger = logging.getLogger(__name__)
    result = feedback_analyzer.analyze_customer_feedback(test_feedback)
    logger.info(f"Sentiment: {result['sentiment']}")
    logger.info(f"Category: {result['category']}")
