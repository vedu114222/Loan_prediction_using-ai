from pydantic import BaseModel, Field


class LoanRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    salary: float = Field(..., gt=0, description="Annual gross salary in USD")
    credit_score: int = Field(..., ge=300, le=850, description="FICO credit score")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 34,
                    "salary": 95000,
                    "credit_score": 720,
                    "loan_amount": 250000,
                }
            ]
        }
    }
