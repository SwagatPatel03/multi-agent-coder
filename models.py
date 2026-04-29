from enum import Enum

from pydantic import BaseModel, Field


class Subtask(BaseModel):
    id: str = Field(..., description="Unique identifier for the subtask")
    description: str = Field(..., description="Detailed instructions for the worker")
    agent: str = Field(
        ..., description="Which agent handles this: 'coder', 'tester', or 'docs'"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of subtasks that must complete first"
    )
    file_path: str = Field(..., description="The primary file this subtask modifies")


class TaskPlan(BaseModel):
    goal: str = Field(..., description="The overarching user request")
    subtasks: list[Subtask] = Field(..., description="The DAG of subtasks")
    language: str = Field(default="python", description="Target programming language")
    framework: str | None = Field(
        default=None, description="Target framework, e.g., FastAPI"
    )


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    REJECT = "REJECT"


class ReviewIssue(BaseModel):
    id: str = Field(..., description="Unique identifier for issue")
    description: str
    severity: str = Field(..., description="'blocking' or 'suggestion'")


class ReviewResult(BaseModel):
    decision: ReviewDecision
    issues: list[ReviewIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    score: int = Field(..., ge=0, le=100)


class ExecutionResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class TestFailure(BaseModel):
    name: str
    message: str


class TestOutcome(BaseModel):
    passed: int
    total: int
    failures: list[TestFailure] = Field(default_factory=list)


class AgentOutputs(BaseModel):
    code: str = ""
    tests: str = ""
    docs: str = ""
