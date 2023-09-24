from vllm.outputs import RequestOutput, CompletionOutput

class CompletionOutputV2(CompletionOutput):
    @classmethod
    def from_v1_object(cls, v1_obj):
        return cls(
            index = v1_obj.index,
            text = v1_obj.text,
            token_ids = v1_obj.token_ids,
            cumulative_logprob = v1_obj.cumulative_logprob,
            logprobs = v1_obj.logprobs,
            finish_reason = v1_obj.finish_reason,
        )
        
    def to_json(self) -> str:
        return {
            "index": self.index,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "logprobs": self.logprobs,
            "cumulative_logprob": self.cumulative_logprob,
            "finish_reason": repr(self.finish_reason),
        }

class RequestOutputV2(RequestOutput):
    @classmethod
    def from_v1_object(cls, v1_obj):
        return cls(
            request_id=v1_obj.request_id,
            prompt=v1_obj.prompt,
            prompt_token_ids=v1_obj.prompt_token_ids,
            outputs=v1_obj.outputs,
            finished = v1_obj.finished,
        )

    def to_json(self) -> str:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "prompt_token_ids": list(self.prompt_token_ids),
            "outputs": [CompletionOutputV2.from_v1_object(completion_output).to_json() for completion_output in self.outputs],
        }