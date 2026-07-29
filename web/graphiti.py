#!/usr/bin/env python3
"""
The Graphiti memory graph, read over Neo4j's HTTP query endpoint.

Graphiti stores the agent memory this stack accumulates — episodes, the entities
extracted from them, and the relationships between those entities — in Neo4j.
The manager only ever reads it, so this is a query layer rather than a client:
Cypher in, plain dictionaries out, with the shapes the UI panels expect.

Two things here are less obvious than they look.

`neo4j_rows_as_dicts` exists because the HTTP API answers with columns and rows
as parallel arrays rather than records, and every caller would otherwise re-zip
them. `normalize_entity_labels` exists because Graphiti labels every node
`Entity` in addition to its specific type, so the useful label is whatever is
left after that one is removed.

Relationship and episode queries carry explicit `limit` parameters rather than
returning everything: this graph grows without bound as the agent runs, and the
panels that read it are previews.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

import config_env
import core


def graphiti_config(env: dict | None = None) -> dict:
    env = env or config_env.read_env()
    graphiti_url = (
        env.get('GRAPHITI_PUBLIC_URL')
        or f"http://127.0.0.1:{env.get('GRAPHITI_PORT', '8070')}"
    ).rstrip('/')
    parsed_bolt = urlparse(env.get('GRAPHITI_NEO4J_URI', 'bolt://127.0.0.1:7687'))
    neo4j_host_default = parsed_bolt.hostname or env.get('GRAPHITI_NEO4J_BOLT_BIND', '127.0.0.1')
    neo4j_http_host = env.get('GRAPHITI_NEO4J_HTTP_BIND', neo4j_host_default)
    neo4j_http_port = env.get('GRAPHITI_NEO4J_HTTP_PORT', '7474')
    return {
        'graphiti_url': graphiti_url,
        'llm_base_url': (env.get('GRAPHITI_LLM_BASE_URL') or '').rstrip('/'),
        'llm_model': env.get('GRAPHITI_LLM_MODEL', ''),
        'embed_base_url': (env.get('GRAPHITI_EMBED_BASE_URL') or '').rstrip('/'),
        'embed_model': env.get('GRAPHITI_EMBED_MODEL', ''),
        'reranker_provider': env.get('GRAPHITI_RERANKER_PROVIDER', ''),
        'reranker_base_url': (env.get('GRAPHITI_RERANKER_BASE_URL') or '').rstrip('/'),
        'reranker_model': env.get('GRAPHITI_RERANKER_MODEL', ''),
        'neo4j_uri': env.get('GRAPHITI_NEO4J_URI', ''),
        'neo4j_user': env.get('GRAPHITI_NEO4J_USER', ''),
        'neo4j_password': env.get('GRAPHITI_NEO4J_PASSWORD', ''),
        'neo4j_database': env.get('GRAPHITI_NEO4J_DATABASE', 'neo4j'),
        'neo4j_http_url': f"http://{neo4j_http_host}:{neo4j_http_port}",
    }


def neo4j_http_query(cypher: str, parameters: dict | None = None, timeout: int = 15) -> dict:
    cfg = graphiti_config()
    statement = {'statement': cypher, 'parameters': parameters or {}}
    payload = {'statements': [statement]}
    auth_bytes = f"{cfg['neo4j_user']}:{cfg['neo4j_password']}".encode('utf-8')
    auth_header = base64.b64encode(auth_bytes).decode('ascii')
    req = urlrequest.Request(
        f"{cfg['neo4j_http_url']}/db/{cfg['neo4j_database']}/tx/commit",
        data=json.dumps(payload).encode('utf-8'),
        method='POST',
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Basic {auth_header}',
        },
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    data = json.loads(body.decode('utf-8')) if body else {}
    errors = data.get('errors') or []
    if errors:
        raise RuntimeError(errors[0].get('message', 'Neo4j query failed'))
    results = data.get('results') or []
    if not results:
        return {'columns': [], 'rows': []}
    rows = []
    columns = results[0].get('columns', [])
    for item in results[0].get('data', []):
        rows.append(item.get('row', []))
    return {'columns': columns, 'rows': rows}


def neo4j_rows_as_dicts(result: dict) -> list[dict]:
    columns = result.get('columns', [])
    out = []
    for row in result.get('rows', []):
        out.append({columns[i]: row[i] for i in range(min(len(columns), len(row)))})
    return out


def normalize_entity_labels(raw_labels) -> list[str]:
    if isinstance(raw_labels, list):
        return [str(x) for x in raw_labels if str(x) and str(x) != 'Entity']
    return []


def graphiti_recent_episodes(
    page: int = 1,
    page_size: int = 25,
    group_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    skip = (page - 1) * page_size
    where = []
    params: dict[str, object] = {'skip': skip, 'limit': page_size}
    if group_id:
        where.append('e.group_id = $group_id')
        params['group_id'] = group_id
    if start_time:
        where.append('e.created_at >= datetime($start_time)')
        params['start_time'] = start_time
    if end_time:
        where.append('e.created_at <= datetime($end_time)')
        params['end_time'] = end_time
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    count_query = f"MATCH (e:Episodic) {where_clause} RETURN count(e) AS total"
    list_query = f"""
        MATCH (e:Episodic)
        {where_clause}
        RETURN e.uuid AS uuid,
               e.name AS name,
               e.group_id AS group_id,
               toString(e.created_at) AS created_at,
               toString(e.valid_at) AS valid_at,
               e.source AS source,
               e.source_description AS source_description,
               e.content AS content
        ORDER BY e.created_at DESC
        SKIP $skip
        LIMIT $limit
    """
    total_rows = neo4j_rows_as_dicts(neo4j_http_query(count_query, params))
    total = int(total_rows[0]['total']) if total_rows else 0
    rows = neo4j_rows_as_dicts(neo4j_http_query(list_query, params))
    items = []
    for row in rows:
        items.append(
            {
                'uuid': row.get('uuid'),
                'name': row.get('name') or '',
                'group_id': row.get('group_id') or '',
                'created_at': row.get('created_at'),
                'valid_at': row.get('valid_at'),
                'source': row.get('source') or '',
                'source_description': row.get('source_description') or '',
                'content_snippet': core.truncate_text(row.get('content'), 280),
                'content': row.get('content') or '',
            }
        )
    return {'items': items, 'page': page, 'page_size': page_size, 'total': total}


def graphiti_recent_entities(
    page: int = 1,
    page_size: int = 25,
    group_id: str | None = None,
    name_query: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    skip = (page - 1) * page_size
    where = []
    params: dict[str, object] = {'skip': skip, 'limit': page_size}
    if group_id:
        where.append('n.group_id = $group_id')
        params['group_id'] = group_id
    if name_query:
        where.append('toLower(coalesce(n.name, "")) CONTAINS toLower($name_query)')
        params['name_query'] = name_query
    if start_time:
        where.append('n.created_at >= datetime($start_time)')
        params['start_time'] = start_time
    if end_time:
        where.append('n.created_at <= datetime($end_time)')
        params['end_time'] = end_time
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    count_query = f"MATCH (n:Entity) {where_clause} RETURN count(n) AS total"
    list_query = f"""
        MATCH (n:Entity)
        {where_clause}
        RETURN n.uuid AS uuid,
               n.name AS name,
               n.group_id AS group_id,
               toString(n.created_at) AS created_at,
               n.summary AS summary,
               n.labels AS prop_labels,
               [x IN labels(n) WHERE x <> 'Entity'] AS node_labels,
               COUNT {{ (n)--() }} AS degree
        ORDER BY n.created_at DESC
        SKIP $skip
        LIMIT $limit
    """
    total_rows = neo4j_rows_as_dicts(neo4j_http_query(count_query, params))
    total = int(total_rows[0]['total']) if total_rows else 0
    rows = neo4j_rows_as_dicts(neo4j_http_query(list_query, params))
    items = []
    for row in rows:
        labels = normalize_entity_labels(row.get('prop_labels')) or normalize_entity_labels(row.get('node_labels'))
        items.append(
            {
                'uuid': row.get('uuid'),
                'name': row.get('name') or '',
                'group_id': row.get('group_id') or '',
                'created_at': row.get('created_at'),
                'summary': row.get('summary') or '',
                'summary_snippet': core.truncate_text(row.get('summary'), 240),
                'labels': labels,
                'degree': int(row.get('degree') or 0),
            }
        )
    return {'items': items, 'page': page, 'page_size': page_size, 'total': total}


def graphiti_recent_relationships(
    page: int = 1,
    page_size: int = 25,
    group_id: str | None = None,
    relation_query: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    skip = (page - 1) * page_size
    where = []
    params: dict[str, object] = {'skip': skip, 'limit': page_size}
    if group_id:
        where.append('r.group_id = $group_id')
        params['group_id'] = group_id
    if relation_query:
        where.append('toLower(coalesce(r.name, "")) CONTAINS toLower($relation_query)')
        params['relation_query'] = relation_query
    if start_time:
        where.append('r.created_at >= datetime($start_time)')
        params['start_time'] = start_time
    if end_time:
        where.append('r.created_at <= datetime($end_time)')
        params['end_time'] = end_time
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    count_query = f"MATCH (:Entity)-[r:RELATES_TO]->(:Entity) {where_clause} RETURN count(r) AS total"
    list_query = f"""
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        {where_clause}
        RETURN r.uuid AS uuid,
               r.group_id AS group_id,
               r.name AS relation_name,
               r.fact AS fact,
               toString(r.created_at) AS created_at,
               toString(r.valid_at) AS valid_at,
               toString(r.invalid_at) AS invalid_at,
               toString(r.expired_at) AS expired_at,
               s.uuid AS source_uuid,
               s.name AS source_name,
               t.uuid AS target_uuid,
               t.name AS target_name
        ORDER BY r.created_at DESC
        SKIP $skip
        LIMIT $limit
    """
    total_rows = neo4j_rows_as_dicts(neo4j_http_query(count_query, params))
    total = int(total_rows[0]['total']) if total_rows else 0
    rows = neo4j_rows_as_dicts(neo4j_http_query(list_query, params))
    items = []
    for row in rows:
        items.append(
            {
                'uuid': row.get('uuid'),
                'group_id': row.get('group_id') or '',
                'relation_name': row.get('relation_name') or '',
                'fact': row.get('fact') or '',
                'fact_snippet': core.truncate_text(row.get('fact'), 260),
                'created_at': row.get('created_at'),
                'valid_at': row.get('valid_at'),
                'invalid_at': row.get('invalid_at'),
                'expired_at': row.get('expired_at'),
                'source_uuid': row.get('source_uuid'),
                'source_name': row.get('source_name') or '',
                'target_uuid': row.get('target_uuid'),
                'target_name': row.get('target_name') or '',
            }
        )
    return {'items': items, 'page': page, 'page_size': page_size, 'total': total}


def graphiti_markdown_export(payload: dict) -> str:
    meta = payload.get('metadata', {})
    lines = [
        f"# Graphiti Export: {meta.get('export_type', 'unknown')}",
        '',
        f"- Exported at: `{meta.get('exported_at', '')}`",
        f"- Graphiti URL: `{meta.get('graphiti_url', '')}`",
        f"- Neo4j DB: `{meta.get('neo4j_database', '')}`",
        f"- Item count: `{meta.get('item_count', 0)}`",
        '',
    ]

    if payload.get('episodes'):
        lines.append('## Episodes')
        for ep in payload['episodes']:
            lines.extend(
                [
                    f"### {ep.get('name') or ep.get('uuid')}",
                    f"- UUID: `{ep.get('uuid')}`",
                    f"- Group: `{ep.get('group_id', '')}`",
                    f"- Created: `{ep.get('created_at', '')}`",
                    f"- Source: `{ep.get('source', '')}`",
                    f"- Source Description: `{ep.get('source_description', '')}`",
                    '',
                    core.truncate_text(ep.get('content', ''), 2000) or '_No content_',
                    '',
                ]
            )

    if payload.get('entities'):
        lines.append('## Entities')
        for ent in payload['entities']:
            labels = ', '.join(ent.get('labels') or [])
            lines.extend(
                [
                    f"### {ent.get('name') or ent.get('uuid')}",
                    f"- UUID: `{ent.get('uuid')}`",
                    f"- Group: `{ent.get('group_id', '')}`",
                    f"- Created: `{ent.get('created_at', '')}`",
                    f"- Labels: `{labels}`",
                    f"- Degree: `{ent.get('degree', 0)}`",
                    f"- Summary: {core.truncate_text(ent.get('summary', ''), 600)}",
                    '',
                ]
            )

    if payload.get('relationships'):
        lines.append('## Relationships')
        for rel in payload['relationships']:
            lines.extend(
                [
                    f"### {rel.get('relation_name') or rel.get('uuid')}",
                    f"- UUID: `{rel.get('uuid')}`",
                    f"- Group: `{rel.get('group_id', '')}`",
                    f"- Created: `{rel.get('created_at', '')}`",
                    f"- Source: `{rel.get('source_name', '')}` (`{rel.get('source_uuid', '')}`)",
                    f"- Target: `{rel.get('target_name', '')}` (`{rel.get('target_uuid', '')}`)",
                    f"- Fact: {core.truncate_text(rel.get('fact', ''), 700)}",
                    '',
                ]
            )

    return '\n'.join(lines).strip() + '\n'


def safe_export_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', name)


def graphiti_entity_neighborhood(entity_uuid: str, limit: int = 50) -> dict:
    query = """
        MATCH (center:Entity {uuid: $uuid})
        OPTIONAL MATCH (center)-[r1:RELATES_TO]->(n1:Entity)
        WITH center, collect(DISTINCT {
          uuid: r1.uuid,
          relation_name: r1.name,
          fact: r1.fact,
          direction: 'out',
          source_uuid: center.uuid,
          source_name: center.name,
          target_uuid: n1.uuid,
          target_name: n1.name
        }) AS outgoing
        OPTIONAL MATCH (n2:Entity)-[r2:RELATES_TO]->(center)
        WITH center, outgoing, collect(DISTINCT {
          uuid: r2.uuid,
          relation_name: r2.name,
          fact: r2.fact,
          direction: 'in',
          source_uuid: n2.uuid,
          source_name: n2.name,
          target_uuid: center.uuid,
          target_name: center.name
        }) AS incoming
        RETURN center.uuid AS uuid,
               center.name AS name,
               center.group_id AS group_id,
               toString(center.created_at) AS created_at,
               center.summary AS summary,
               outgoing[0..$limit] AS outgoing,
               incoming[0..$limit] AS incoming
    """
    rows = neo4j_rows_as_dicts(neo4j_http_query(query, {'uuid': entity_uuid, 'limit': limit}))
    if not rows:
        raise RuntimeError('Entity not found')
    return rows[0]
