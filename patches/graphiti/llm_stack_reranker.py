from __future__ import annotations

import logging

import httpx

from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)


class LlamaCppRerankerClient(CrossEncoderClient):
    """
    Cross-encoder adapter for llama.cpp's /v1/rerank endpoint.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key or ''

    def _rerank_url(self) -> str:
        if self.base_url.endswith('/v1'):
            return f'{self.base_url}/rerank'
        return f'{self.base_url}/v1/rerank'

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        headers = {}
        if self.api_key and self.api_key.lower() not in {'none', 'null'}:
            headers['Authorization'] = f'Bearer {self.api_key}'

        payload = {
            'model': self.model,
            'query': query,
            'documents': passages,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self._rerank_url(), json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        # llama.cpp style response:
        # {"results":[{"index":0,"relevance_score":0.99}, ...]}
        results = data.get('results', data.get('data', []))
        scored_passages: list[tuple[str, float]] = []
        for idx, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            passage_index = item.get('index', idx)
            if not isinstance(passage_index, int):
                continue
            if passage_index < 0 or passage_index >= len(passages):
                continue
            score = item.get('relevance_score', item.get('score', 0.0))
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = 0.0
            scored_passages.append((passages[passage_index], numeric_score))

        if not scored_passages:
            logger.warning('Reranker returned no usable scores, falling back to input order.')
            return [(passage, 0.0) for passage in passages]

        scored_passages.sort(key=lambda x: x[1], reverse=True)
        return scored_passages
