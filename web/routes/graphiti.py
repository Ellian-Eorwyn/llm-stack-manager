#!/usr/bin/env python3
"""
HTTP routes for the Graphiti memory panel.

Thin wrappers over `graphiti.py`: parse the query string, call a query, return
JSON. The one exception is the export, which renders Markdown to a file under
`core.GRAPHITI_EXPORTS_DIR` and then serves it back — a browser cannot stream a
multi-megabyte graph dump into a download any other way.

Registered without a `url_prefix` so the rules stay exactly what they were when
these lived in `app.py`; `tests/test_llm_stack_manager.RouteInventoryTests`
holds them to that.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request, send_file

import config_env
import core
import graphiti

bp = Blueprint("graphiti", __name__)


@bp.route('/api/graphiti/status')
def api_graphiti_status():
    env = config_env.read_env()
    cfg = graphiti.graphiti_config(env)
    status = {
        'ok': True,
        'last_refresh': int(time.time()),
        'config': {
            'graphiti_url': cfg['graphiti_url'],
            'llm_base_url': cfg['llm_base_url'],
            'llm_model': cfg['llm_model'],
            'embed_base_url': cfg['embed_base_url'],
            'embed_model': cfg['embed_model'],
            'reranker_provider': cfg['reranker_provider'],
            'reranker_base_url': cfg['reranker_base_url'],
            'reranker_model': cfg['reranker_model'],
            'neo4j_uri': cfg['neo4j_uri'],
            'neo4j_database': cfg['neo4j_database'],
            'neo4j_http_url': cfg['neo4j_http_url'],
        },
        'checks': {
            'graphiti_api': {'ok': False, 'error': ''},
            'neo4j': {'ok': False, 'error': ''},
            'llm_endpoint': {'ok': False, 'error': ''},
            'embedding_endpoint': {'ok': False, 'error': ''},
            'reranker_endpoint': {'ok': False, 'error': ''},
            'ingestion_worker': {'ok': None, 'error': 'not exposed by current Graphiti API'},
        },
    }

    try:
        graphiti_health = core.http_json(f"{cfg['graphiti_url']}/healthcheck", timeout=6)
        status['checks']['graphiti_api']['ok'] = graphiti_health.get('status') == 'healthy'
    except Exception as exc:
        status['checks']['graphiti_api']['error'] = str(exc)

    try:
        graphiti.neo4j_http_query('RETURN 1 AS ok', timeout=8)
        status['checks']['neo4j']['ok'] = True
    except Exception as exc:
        status['checks']['neo4j']['error'] = str(exc)

    endpoint_checks = [
        ('llm_endpoint', cfg['llm_base_url']),
        ('embedding_endpoint', cfg['embed_base_url']),
        ('reranker_endpoint', cfg['reranker_base_url']),
    ]
    for key, base_url in endpoint_checks:
        if not base_url:
            status['checks'][key]['error'] = 'not configured'
            continue
        try:
            core.http_json(f"{base_url}/models", timeout=6)
            status['checks'][key]['ok'] = True
        except Exception as exc:
            status['checks'][key]['error'] = str(exc)

    status['ok'] = all(v.get('ok') is True for k, v in status['checks'].items() if k != 'ingestion_worker')
    return jsonify(status)


@bp.route('/api/graphiti/stats')
def api_graphiti_stats():
    try:
        total_counts_query = """
            CALL {
              MATCH (e:Episodic) RETURN count(e) AS episodes
            }
            CALL {
              MATCH (n:Entity) RETURN count(n) AS entities
            }
            CALL {
              MATCH (:Entity)-[r:RELATES_TO]->(:Entity) RETURN count(r) AS relationships
            }
            RETURN episodes, entities, relationships
        """
        counts_rows = graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(total_counts_query))
        counts = counts_rows[0] if counts_rows else {'episodes': 0, 'entities': 0, 'relationships': 0}

        by_day_episodes_query = """
            MATCH (e:Episodic)
            WHERE e.created_at IS NOT NULL
            WITH toString(date(datetime(e.created_at))) AS day, count(e) AS c
            RETURN day, c
            ORDER BY day DESC
            LIMIT 30
        """
        by_day_entities_query = """
            MATCH (n:Entity)
            WHERE n.created_at IS NOT NULL
            WITH toString(date(datetime(n.created_at))) AS day, count(n) AS c
            RETURN day, c
            ORDER BY day DESC
            LIMIT 30
        """
        top_groups_query = """
            MATCH (e:Episodic)
            WHERE coalesce(e.group_id, '') <> ''
            RETURN e.group_id AS group_id, count(e) AS c
            ORDER BY c DESC
            LIMIT 15
        """
        top_entities_query = """
            MATCH (n:Entity)
            WITH n, COUNT { (n)--() } AS degree
            RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id, degree
            ORDER BY degree DESC
            LIMIT 15
        """

        return jsonify(
            {
                'ok': True,
                'totals': {
                    'episodes': int(counts.get('episodes', 0) or 0),
                    'entities': int(counts.get('entities', 0) or 0),
                    'relationships': int(counts.get('relationships', 0) or 0),
                },
                'episodes_by_day': graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(by_day_episodes_query)),
                'entities_by_day': graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(by_day_entities_query)),
                'top_groups': graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(top_groups_query)),
                'top_entities': graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(top_entities_query)),
                'last_refresh': int(time.time()),
            }
        )
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/recent/episodes')
def api_graphiti_recent_episodes():
    page = core.parse_int(request.args.get('page'), 1, minimum=1)
    page_size = core.parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    group_id = (request.args.get('group_id') or '').strip() or None
    start_time = core.parse_iso_datetime(request.args.get('start_time'))
    end_time = core.parse_iso_datetime(request.args.get('end_time'))
    try:
        data = graphiti.graphiti_recent_episodes(
            page=page,
            page_size=page_size,
            group_id=group_id,
            start_time=start_time,
            end_time=end_time,
        )
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/recent/entities')
def api_graphiti_recent_entities():
    page = core.parse_int(request.args.get('page'), 1, minimum=1)
    page_size = core.parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    group_id = (request.args.get('group_id') or '').strip() or None
    name_query = (request.args.get('q') or '').strip() or None
    start_time = core.parse_iso_datetime(request.args.get('start_time'))
    end_time = core.parse_iso_datetime(request.args.get('end_time'))
    try:
        data = graphiti.graphiti_recent_entities(
            page=page,
            page_size=page_size,
            group_id=group_id,
            name_query=name_query,
            start_time=start_time,
            end_time=end_time,
        )
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/recent/relationships')
def api_graphiti_recent_relationships():
    page = core.parse_int(request.args.get('page'), 1, minimum=1)
    page_size = core.parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    group_id = (request.args.get('group_id') or '').strip() or None
    relation_query = (request.args.get('q') or '').strip() or None
    start_time = core.parse_iso_datetime(request.args.get('start_time'))
    end_time = core.parse_iso_datetime(request.args.get('end_time'))
    try:
        data = graphiti.graphiti_recent_relationships(
            page=page,
            page_size=page_size,
            group_id=group_id,
            relation_query=relation_query,
            start_time=start_time,
            end_time=end_time,
        )
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/detail/episode/<episode_uuid>')
def api_graphiti_episode_detail(episode_uuid):
    query = """
        MATCH (e:Episodic {uuid: $uuid})
        OPTIONAL MATCH (e)-[m:MENTIONS]->(n:Entity)
        RETURN e.uuid AS uuid,
               e.name AS name,
               e.group_id AS group_id,
               toString(e.created_at) AS created_at,
               toString(e.valid_at) AS valid_at,
               e.source AS source,
               e.source_description AS source_description,
               e.content AS content,
               collect(DISTINCT {
                 uuid: n.uuid,
                 name: n.name,
                 group_id: n.group_id,
                 mention_uuid: m.uuid
               }) AS entities
    """
    try:
        rows = graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(query, {'uuid': episode_uuid}))
        if not rows:
            return jsonify({'ok': False, 'error': 'Episode not found'}), 404
        return jsonify({'ok': True, 'item': rows[0]})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/detail/entity/<entity_uuid>')
def api_graphiti_entity_detail(entity_uuid):
    query = """
        MATCH (n:Entity {uuid: $uuid})
        CALL {
          WITH n
          OPTIONAL MATCH (ep:Episodic)-[m:MENTIONS]->(n)
          RETURN collect(DISTINCT {
            uuid: ep.uuid,
            name: ep.name,
            group_id: ep.group_id,
            created_at: toString(ep.created_at),
            mention_uuid: m.uuid
          }) AS episodes
        }
        CALL {
          WITH n
          OPTIONAL MATCH (n)-[r:RELATES_TO]->(t:Entity)
          RETURN collect(DISTINCT {
            uuid: r.uuid,
            relation_name: r.name,
            fact: r.fact,
            group_id: r.group_id,
            created_at: toString(r.created_at),
            target_uuid: t.uuid,
            target_name: t.name,
            direction: 'out'
          }) AS outgoing
        }
        CALL {
          WITH n
          OPTIONAL MATCH (s:Entity)-[r:RELATES_TO]->(n)
          RETURN collect(DISTINCT {
            uuid: r.uuid,
            relation_name: r.name,
            fact: r.fact,
            group_id: r.group_id,
            created_at: toString(r.created_at),
            source_uuid: s.uuid,
            source_name: s.name,
            direction: 'in'
          }) AS incoming
        }
        RETURN n.uuid AS uuid,
               n.name AS name,
               n.group_id AS group_id,
               toString(n.created_at) AS created_at,
               n.summary AS summary,
               n.labels AS prop_labels,
               [x IN labels(n) WHERE x <> 'Entity'] AS node_labels,
               episodes,
               outgoing,
               incoming
    """
    try:
        rows = graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(query, {'uuid': entity_uuid}))
        if not rows:
            return jsonify({'ok': False, 'error': 'Entity not found'}), 404
        item = rows[0]
        item['labels'] = graphiti.normalize_entity_labels(item.get('prop_labels')) or graphiti.normalize_entity_labels(item.get('node_labels'))
        return jsonify({'ok': True, 'item': item})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/detail/relationship/<relationship_uuid>')
def api_graphiti_relationship_detail(relationship_uuid):
    query = """
        MATCH (s:Entity)-[r:RELATES_TO {uuid: $uuid}]->(t:Entity)
        OPTIONAL MATCH (ep:Episodic)
        WHERE r.episodes IS NOT NULL AND ep.uuid IN r.episodes
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
               t.name AS target_name,
               collect(DISTINCT {
                 uuid: ep.uuid,
                 name: ep.name,
                 group_id: ep.group_id,
                 created_at: toString(ep.created_at)
               }) AS linked_episodes
    """
    try:
        rows = graphiti.neo4j_rows_as_dicts(graphiti.neo4j_http_query(query, {'uuid': relationship_uuid}))
        if not rows:
            return jsonify({'ok': False, 'error': 'Relationship not found'}), 404
        return jsonify({'ok': True, 'item': rows[0]})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/search/memory', methods=['POST'])
def api_graphiti_search_memory():
    data = request.json or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({'ok': False, 'error': 'query is required'}), 400
    group_id = (data.get('group_id') or '').strip()
    max_facts = core.parse_int(data.get('max_facts'), 10, minimum=1, maximum=50)
    cfg = graphiti.graphiti_config()
    payload = {'query': query, 'max_facts': max_facts}
    if group_id:
        payload['group_ids'] = [group_id]
    try:
        result = core.http_json(f"{cfg['graphiti_url']}/search", method='POST', payload=payload, timeout=20)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/search/group/<group_id>')
def api_graphiti_group_history(group_id):
    last_n = core.parse_int(request.args.get('last_n'), 50, minimum=1, maximum=500)
    cfg = graphiti.graphiti_config()
    try:
        result = core.http_json(f"{cfg['graphiti_url']}/episodes/{group_id}?last_n={last_n}", timeout=25)
        return jsonify({'ok': True, 'group_id': group_id, 'episodes': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/search/entities')
def api_graphiti_search_entities():
    q = (request.args.get('q') or '').strip()
    page = core.parse_int(request.args.get('page'), 1, minimum=1)
    page_size = core.parse_int(request.args.get('page_size'), 25, minimum=1, maximum=100)
    try:
        data = graphiti.graphiti_recent_entities(page=page, page_size=page_size, name_query=q or None)
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/neighborhood/<entity_uuid>')
def api_graphiti_neighborhood(entity_uuid):
    limit = core.parse_int(request.args.get('limit'), 50, minimum=1, maximum=200)
    try:
        item = graphiti.graphiti_entity_neighborhood(entity_uuid, limit=limit)
        return jsonify({'ok': True, 'item': item})
    except RuntimeError as exc:
        if str(exc) == 'Entity not found':
            return jsonify({'ok': False, 'error': 'Entity not found'}), 404
        return jsonify({'ok': False, 'error': str(exc)}), 500
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@bp.route('/api/graphiti/exports')
def api_graphiti_exports_list():
    files = []
    for path in sorted(core.GRAPHITI_EXPORTS_DIR.glob('*'), reverse=True):
        if not path.is_file():
            continue
        files.append(
            {
                'filename': path.name,
                'size_bytes': path.stat().st_size,
                'modified_at': int(path.stat().st_mtime),
                'download_url': f"/api/graphiti/exports/{path.name}",
            }
        )
    return jsonify({'ok': True, 'items': files[:200], 'directory': str(core.GRAPHITI_EXPORTS_DIR)})


@bp.route('/api/graphiti/exports/<path:filename>')
def api_graphiti_export_download(filename):
    safe_name = graphiti.safe_export_filename(filename)
    path = core.GRAPHITI_EXPORTS_DIR / safe_name
    if not path.exists() or not path.is_file():
        return jsonify({'ok': False, 'error': 'Export file not found'}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@bp.route('/api/graphiti/export', methods=['POST'])
def api_graphiti_export():
    data = request.json or {}
    export_type = (data.get('export_type') or 'recent').strip()
    fmt = (data.get('format') or 'json').strip().lower()
    if fmt not in ('json', 'md'):
        return jsonify({'ok': False, 'error': 'format must be json or md'}), 400

    env = config_env.read_env()
    cfg = graphiti.graphiti_config(env)
    timestamp = int(time.time())
    dt = datetime.utcfromtimestamp(timestamp).strftime('%Y%m%d-%H%M%S')
    payload: dict = {
        'metadata': {
            'export_type': export_type,
            'exported_at': datetime.utcfromtimestamp(timestamp).isoformat() + 'Z',
            'graphiti_url': cfg['graphiti_url'],
            'neo4j_database': cfg['neo4j_database'],
        },
        'episodes': [],
        'entities': [],
        'relationships': [],
    }

    try:
        if export_type == 'group':
            group_id = (data.get('group_id') or '').strip()
            if not group_id:
                return jsonify({'ok': False, 'error': 'group_id is required for export_type=group'}), 400
            limit = core.parse_int(data.get('limit'), 200, minimum=1, maximum=5000)
            payload['metadata']['group_id'] = group_id
            payload['episodes'] = graphiti.graphiti_recent_episodes(page=1, page_size=limit, group_id=group_id)['items']
            payload['entities'] = graphiti.graphiti_recent_entities(page=1, page_size=limit, group_id=group_id)['items']
            payload['relationships'] = graphiti.graphiti_recent_relationships(page=1, page_size=limit, group_id=group_id)['items']
            base_name = f"graphiti-group-{graphiti.safe_export_filename(group_id)}-{dt}"
        elif export_type == 'entity':
            entity_uuid = (data.get('entity_uuid') or '').strip()
            if not entity_uuid:
                return jsonify({'ok': False, 'error': 'entity_uuid is required for export_type=entity'}), 400
            entity_rows = graphiti.neo4j_rows_as_dicts(
                graphiti.neo4j_http_query(
                    """
                    MATCH (n:Entity {uuid: $uuid})
                    RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id,
                           toString(n.created_at) AS created_at, n.summary AS summary,
                           n.labels AS prop_labels, [x IN labels(n) WHERE x <> 'Entity'] AS node_labels,
                           COUNT { (n)--() } AS degree
                    """,
                    {'uuid': entity_uuid},
                )
            )
            if not entity_rows:
                return jsonify({'ok': False, 'error': 'Entity not found'}), 404
            ent = entity_rows[0]
            ent['labels'] = graphiti.normalize_entity_labels(ent.get('prop_labels')) or graphiti.normalize_entity_labels(ent.get('node_labels'))
            payload['entities'] = [ent]
            neighborhood = graphiti.graphiti_entity_neighborhood(entity_uuid, limit=200)
            rels = (neighborhood.get('outgoing') or []) + (neighborhood.get('incoming') or [])
            payload['relationships'] = rels
            payload['metadata']['entity_uuid'] = entity_uuid
            base_name = f"graphiti-entity-{graphiti.safe_export_filename(entity_uuid)}-{dt}"
        elif export_type == 'recent':
            limit = core.parse_int(data.get('limit'), 200, minimum=1, maximum=5000)
            payload['episodes'] = graphiti.graphiti_recent_episodes(page=1, page_size=limit)['items']
            payload['entities'] = graphiti.graphiti_recent_entities(page=1, page_size=limit)['items']
            payload['relationships'] = graphiti.graphiti_recent_relationships(page=1, page_size=limit)['items']
            payload['metadata']['limit'] = limit
            base_name = f"graphiti-recent-{limit}-{dt}"
        elif export_type == 'date_range':
            start_time = core.parse_iso_datetime(data.get('start_time'))
            end_time = core.parse_iso_datetime(data.get('end_time'))
            if not start_time or not end_time:
                return jsonify({'ok': False, 'error': 'start_time and end_time are required for date_range export'}), 400
            limit = core.parse_int(data.get('limit'), 1000, minimum=1, maximum=10000)
            payload['episodes'] = graphiti.graphiti_recent_episodes(page=1, page_size=limit, start_time=start_time, end_time=end_time)['items']
            payload['entities'] = graphiti.graphiti_recent_entities(page=1, page_size=limit, start_time=start_time, end_time=end_time)['items']
            payload['relationships'] = graphiti.graphiti_recent_relationships(page=1, page_size=limit, start_time=start_time, end_time=end_time)['items']
            payload['metadata']['start_time'] = start_time
            payload['metadata']['end_time'] = end_time
            payload['metadata']['limit'] = limit
            base_name = f"graphiti-date-range-{dt}"
        else:
            return jsonify({'ok': False, 'error': f'unsupported export_type: {export_type}'}), 400

        payload['metadata']['item_count'] = (
            len(payload.get('episodes', []))
            + len(payload.get('entities', []))
            + len(payload.get('relationships', []))
        )

        if fmt == 'json':
            file_name = f"{base_name}.json"
            file_path = core.GRAPHITI_EXPORTS_DIR / file_name
            file_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        else:
            file_name = f"{base_name}.md"
            file_path = core.GRAPHITI_EXPORTS_DIR / file_name
            file_path.write_text(graphiti.graphiti_markdown_export(payload), encoding='utf-8')

        return jsonify(
            {
                'ok': True,
                'file': {
                    'filename': file_name,
                    'path': str(file_path),
                    'size_bytes': file_path.stat().st_size,
                    'download_url': f"/api/graphiti/exports/{file_name}",
                },
                'directory': str(core.GRAPHITI_EXPORTS_DIR),
                'summary': payload['metadata'],
            }
        )
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500

