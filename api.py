from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import io
import tempfile
import traceback
from typing import List, Optional, Dict, Any

from core.pipeline import HopfieldQAPipeline, PipelineConfig
from core.llm_provider import LLMConfig, create_llm_provider, MockLLMProvider
from core.chat_manager import ChatManager
from utils.file_utils import extract_text_from_file
from utils.image_extractor import extract_images_from_file_bytes
from pathlib import Path

app = FastAPI(title="Hopfield QA API")

pipeline = None
chat_manager = ChatManager("data")

class SearchRequest(BaseModel):
    query: str
    energy_type: Optional[str] = "softmax"
    hopfield_beta: Optional[float] = 8.0
    top_k: Optional[int] = 5
    use_projection: Optional[bool] = False
    retriever_mode: Optional[str] = "hopfield"  # "hopfield", "dense", "hybrid"
    engine_mode: Optional[str] = "associative"  # legacy backwards compatibility
    answer_style: Optional[str] = "conversational"  # "conversational" (LLM RAG), "detailed", "executive", "concise"
    detail_level: Optional[str] = None  # "adaptive", "detailed", "standard", "concise"
    llm_provider: Optional[str] = None  # "ollama", "openai", "mock", "deterministic"
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    max_tokens: Optional[int] = 8192
    chat_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    web_search_enabled: Optional[bool] = None
    force_web_search: Optional[bool] = False
    answer_format: Optional[str] = None
    active_doc_id: Optional[str] = None

class CreateChatRequest(BaseModel):
    title: Optional[str] = None
    chat_id: Optional[str] = None

class UpdateChatRequest(BaseModel):
    title: str

@app.on_event("startup")
def startup_event():
    global pipeline
    proj_path = "models/projection_squad.pt" if os.path.exists("models/projection_squad.pt") else None
    config = PipelineConfig(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=64,
        hopfield_beta=8.0,
        top_k=5,
        use_fusion=True,
        use_projection=False,
        projection_path=proj_path,
        use_associative_engine=True,
    )
    pipeline = HopfieldQAPipeline(config)

@app.get("/api/status")
def get_status():
    global pipeline
    if not pipeline:
        return {"ready": False, "documents": 0, "chunks": 0, "llm": {"available": False}}

    is_avail = False
    avail_reason = None
    if hasattr(pipeline.answer_generator, "llm_provider") and pipeline.answer_generator.llm_provider:
        is_avail, avail_reason = pipeline.answer_generator.llm_provider.is_available()

    llm_status = {
        "provider": pipeline.config.llm_provider,
        "configured_provider": pipeline.config.llm_provider,
        "effective_provider": pipeline.config.llm_provider,
        "model": pipeline.config.llm_model,
        "base_url": pipeline.config.llm_base_url,
        "fallback_to_deterministic": pipeline.config.llm_fallback_to_deterministic,
        "fallback_enabled": pipeline.config.llm_fallback_to_deterministic,
        "available": is_avail,
        "is_healthy": is_avail,
        "status_message": avail_reason or "Provider is healthy and reachable.",
        "fallback_reason": avail_reason if not is_avail else None
    }

    return {
        "ready": pipeline.is_ready,
        "documents": pipeline.n_documents,
        "chunks": pipeline.n_chunks,
        "llm": llm_status,
    }

@app.post("/api/search")
def search(req: SearchRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    pipeline.config.energy_type = req.energy_type
    pipeline.config.hopfield_beta = req.hopfield_beta
    pipeline.config.top_k = req.top_k or 5
    pipeline.config.use_projection = req.use_projection
    pipeline.memory.retriever.beta = req.hopfield_beta
    pipeline.memory.retriever.energy_type = req.energy_type
    pipeline.memory.retriever.use_projection = req.use_projection
    
    # Determine retriever mode
    active_retriever = (req.retriever_mode or "hopfield").lower()
    if req.engine_mode == "naive_rag" and req.retriever_mode == "hopfield":
        active_retriever = "dense"
        
    style = req.detail_level or req.answer_style or "conversational"

    # Optional dynamic LLM provider override
    provider_override = None
    if req.llm_provider:
        p_name = req.llm_provider.lower().strip()
        ovr_cfg = LLMConfig.from_env()
        ovr_cfg.provider = p_name
        if req.llm_model:
            ovr_cfg.model = req.llm_model
        if req.llm_base_url:
            ovr_cfg.base_url = req.llm_base_url
        if req.llm_api_key:
            ovr_cfg.api_key = req.llm_api_key
        if req.max_tokens:
            ovr_cfg.max_tokens = req.max_tokens
        provider_override = create_llm_provider(ovr_cfg)
    
    # Extract conversation history for multi-turn conversational context
    history = req.conversation_history
    if not history and req.chat_id:
        chat_sess = chat_manager.get_session(req.chat_id)
        if chat_sess:
            history = [{"role": m["role"], "content": m["content"]} for m in chat_sess.get("messages", [])]

    # Adaptive retrieval depth: scale k for detailed, multi-aspect, or explanatory questions
    q_lower = req.query.lower()
    q_words = req.query.strip().split()
    is_multi_aspect = any(w in q_lower for w in [
        "explain", "architecture", "components", "mechanism", "mechanisms", "responsibilities",
        "difference", "compare", "breakdown", "elaborate", "comprehensive", "lifecycle",
        "and how", "and what", "overview", "workflow", "workloads", "detail", "detailed",
        "how does", "why does"
    ]) or len(q_words) >= 6

    user_k = req.top_k
    if user_k and user_k > 5:
        effective_k = user_k
    elif style in ("detailed", "comprehensive", "technical") or is_multi_aspect:
        effective_k = 12
    elif style == "executive" or any(w in q_lower for w in ["what is", "define", "meaning", "concept"]):
        effective_k = 8
    else:
        effective_k = 6

    try:
        res = pipeline.query(
            raw_query=req.query,
            top_k=effective_k,
            retriever_mode=active_retriever,
            answer_style=style,
            llm_provider_override=provider_override,
            conversation_history=history,
            web_search_enabled=req.web_search_enabled,
            force_web_search=bool(req.force_web_search),
            answer_format=req.answer_format,
            active_doc_id=req.active_doc_id,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    
    evidence_list = []
    source_chunks = res.evidence_collection.chunks if (res.evidence_collection and res.evidence_collection.chunks) else res.retrieved_chunks
    for i, chunk in enumerate(source_chunks, 1):
        sim = getattr(chunk, 'similarity_score', getattr(chunk, 'hopfield_attention', 0.0))
        meta = getattr(chunk, 'metadata', {}) or {}
        is_core = getattr(chunk, 'is_core_attractor', meta.get("is_core_attractor", True))
        evidence_list.append({
            "citation_id": f"E{i}",
            "chunk_id": getattr(chunk, 'chunk_id', f"chunk_{i}"),
            "text": getattr(chunk, 'text', ''),
            "score": float(sim) if sim is not None else 0.0,
            "source": getattr(chunk, 'source', 'Unknown'),
            "source_type": getattr(chunk, 'source_type', meta.get('source_type', 'document')),
            "url": getattr(chunk, 'url', meta.get('url')),
            "title": getattr(chunk, 'title', meta.get('title')),
            "published_date": getattr(chunk, 'published_date', meta.get('published_date')),
            "page_number": getattr(chunk, 'page_number', meta.get("page_number")),
            "is_core_attractor": is_core,
            "role": "Core Attractor Basin" if is_core else "Associative Bridge",
            "has_image": meta.get("has_image", False),
            "image_url": meta.get("image_url"),
            "image_caption": meta.get("image_caption"),
            "width": meta.get("width"),
            "height": meta.get("height"),
        })
    
    associative_diagnostics = None
    if res.associative_result:
        ar = res.associative_result
        trajectory_data = []
        for hop in ar.trajectory:
            trajectory_data.append({
                "hop": hop.hop_index,
                "energy": round(hop.energy, 4),
                "delta": round(hop.state_delta, 4),
                "description": hop.description,
                "top_chunks": hop.activated_chunks[:2]
            })
        associative_diagnostics = {
            "mode": f"Hopfield ({res.evidence_collection.retriever_name if res.evidence_collection else 'Associative'})",
            "attractor_energy": round(ar.attractor_energy, 4),
            "energy_depth": round(ar.energy_landscape_depth, 4),
            "is_grounded": ar.is_grounded,
            "convergence_steps": ar.convergence_steps,
            "reasoning_path": ar.reasoning_path,
            "trajectory": trajectory_data,
            "num_core_chunks": len(ar.core_attractor_chunks),
            "num_bridge_chunks": len(ar.associative_bridge_chunks)
        }
    else:
        associative_diagnostics = {
            "mode": f"{res.evidence_collection.retriever_name if res.evidence_collection else 'Dense Cosine'}",
            "attractor_energy": None,
            "energy_depth": None,
            "is_grounded": len(evidence_list) > 0,
            "convergence_steps": 1,
            "reasoning_path": ["Static dot product ranking without continuous associative relaxation."],
            "trajectory": [],
            "num_core_chunks": len(evidence_list),
            "num_bridge_chunks": 0
        }

    grounded_data = None
    if res.grounded_answer:
        ga = res.grounded_answer
        grounded_data = {
            "answer": ga.answer,
            "abstained": ga.abstained,
            "abstention_reason": ga.abstention_reason,
            "confidence": round(ga.confidence, 4),
            "claims": [
                {
                    "claim_text": c.claim_text,
                    "evidence_ids": c.evidence_ids,
                    "is_verified": c.is_verified,
                    "unresolved_ids": c.unresolved_ids,
                    "is_supported": getattr(c, 'is_supported', c.is_verified),
                    "is_relevant": getattr(c, 'is_relevant', True),
                    "is_unique": getattr(c, 'is_unique', True),
                    "verification_rationale": getattr(c, 'verification_rationale', None)
                } for c in ga.claims
            ],
            "provider_used": ga.provider_used,
            "model_name": ga.model_name,
            "fallback_occurred": ga.metadata.get("fallback_occurred", False),
            "fallback_reason": ga.metadata.get("fallback_reason"),
            "llm_config": ga.metadata.get("llm_config", {}),
            "generation_time_ms": max(0.1, round(ga.generation_time_ms, 2)),
            "style": ga.style,
            "stage_latencies_ms": ga.metadata.get("stage_latencies_ms", {}),
            "confidence_breakdown": ga.metadata.get("confidence_breakdown", {}),
            "deduplication": getattr(res, "deduplication_diagnostics", {}) or ga.metadata.get("deduplication", {}),
            "unique_verified_claims_count": ga.metadata.get("unique_verified_claims_count", len([c for c in ga.claims if c.is_verified])),
            "total_claims_count": len(ga.claims)
        }
    
    latencies = {
        "retrieval_ms": round(res.retrieval_time_ms, 2),
        "total_ms": round(res.total_time_ms, 2),
        "stages": getattr(res, "stage_latencies_ms", {})
    }

    contextualized_q = getattr(getattr(res, "processed_query", None), "metadata", {}).get("contextualized_query")

    # Collect relevant visual figures from evidence chunks and grounded_answer
    retrieved_images = []
    seen_images = set()
    for ev in evidence_list:
        img_url = ev.get("image_url")
        if img_url and img_url not in seen_images:
            seen_images.add(img_url)
            retrieved_images.append({
                "image_url": img_url,
                "caption": ev.get("image_caption") or f"Figure from {ev.get('source')}",
                "page_number": ev.get("page_number", 1),
                "source": ev.get("source"),
                "score": ev.get("score", 0.0),
                "width": ev.get("width"),
                "height": ev.get("height"),
            })
    if res.grounded_answer and hasattr(res.grounded_answer, "images") and res.grounded_answer.images:
        for g_img in res.grounded_answer.images:
            img_url = g_img.get("image_url")
            if img_url and img_url not in seen_images:
                seen_images.add(img_url)
                retrieved_images.append(g_img)
    if hasattr(res, "images") and res.images:
        for r_img in res.images:
            img_url = r_img.get("image_url")
            if img_url and img_url not in seen_images:
                seen_images.add(img_url)
                retrieved_images.append(r_img)

    if grounded_data:
        grounded_data["images"] = retrieved_images

    proc_meta = getattr(getattr(res, "processed_query", None), "metadata", {}) or {}
    typos_fixed = proc_meta.get("typos_fixed", {})
    corrected_q = proc_meta.get("corrected_query")

    response_payload = {
        "query": req.query,
        "contextualized_query": contextualized_q,
        "corrected_query": corrected_q,
        "typos_fixed": typos_fixed,
        "retriever_mode": active_retriever,
        "engine_mode": "associative" if active_retriever in ("hopfield", "hybrid") else "naive_rag",
        "answer": res.answer.answer,
        "confidence": float(res.answer.confidence) if res.answer.confidence is not None else 0.0,
        "evidence": evidence_list,
        "images": retrieved_images,
        "grounded_answer": grounded_data,
        "associative_diagnostics": associative_diagnostics,
        "latencies": latencies,
        "deduplication": getattr(res, "deduplication_diagnostics", {}),
        "web_search_used": getattr(res, "web_search_used", False),
        "cached_knowledge_used": getattr(res, "cached_knowledge_used", False),
        "format_detected": getattr(res, "format_detected", None),
        "format_requested": req.answer_format or req.detail_level or req.answer_style,
        "sufficiency_decision": getattr(res, "sufficiency_decision", None),
        "web_sources": getattr(res, "web_sources", [])
    }

    # Automatically persist turn to chat session if chat_id provided
    if req.chat_id:
        chat_sess = chat_manager.add_turn(req.chat_id, req.query, response_payload)
        response_payload["chat_id"] = req.chat_id
        response_payload["chat_title"] = chat_sess.get("title")

    return response_payload


# ==============================================================================
# Multi-Chat Persistence Endpoints (ChatGPT-Style Sessions)
# ==============================================================================

@app.get("/api/chats")
def list_chats():
    """List all persisted chat conversations."""
    return {"chats": chat_manager.list_sessions()}


@app.post("/api/chats")
def create_or_save_chat(req: Optional[CreateChatRequest] = None):
    """Create a new chat conversation."""
    title = req.title if req else None
    c_id = req.chat_id if req else None
    session = chat_manager.create_session(title=title, session_id=c_id)
    return {"chat": session}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    """Retrieve full conversation history for a specific chat."""
    session = chat_manager.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"chat": session}


@app.put("/api/chats/{chat_id}")
def update_chat_title(chat_id: str, req: UpdateChatRequest):
    """Update title of a chat conversation."""
    session = chat_manager.update_session(chat_id, title=req.title)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"chat": session}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    """Delete a chat conversation and remove from disk storage."""
    deleted = chat_manager.delete_session(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"success": True, "message": f"Chat {chat_id} deleted"}


# ==============================================================================
# Persistent Documents / Knowledge Base Endpoints
# ==============================================================================

@app.get("/api/documents")
def list_documents():
    """List all documents currently active and persisted in Hopfield memory."""
    global pipeline
    if not pipeline:
        return {"documents": [], "total_documents": 0, "total_chunks": 0}
        
    chunk_counts = {}
    for c in pipeline._all_chunks:
        chunk_counts[c.source] = chunk_counts.get(c.source, 0) + 1
        
    docs = []
    for source, text in pipeline._documents.items():
        docs.append({
            "source": source,
            "chunks": chunk_counts.get(source, 0),
            "characters": len(text)
        })
    docs.sort(key=lambda d: d["source"].lower())
    return {
        "documents": docs,
        "total_documents": pipeline.n_documents,
        "total_chunks": pipeline.n_chunks
    }


@app.delete("/api/documents/{source:path}")
def remove_document(source: str):
    """Remove a document and its chunks from memory and persist changes."""
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    removed = pipeline.remove_document(source)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{source}' not found in memory")
        
    return {
        "success": True,
        "message": f"Successfully removed '{source}' from Hopfield memory",
        "remaining_documents": pipeline.n_documents,
        "remaining_chunks": pipeline.n_chunks
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB limit
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload limit of 25MB (size: {len(content) / (1024 * 1024):.1f}MB)."
        )

    try:
        ext = Path(file.filename).suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            extracted_images = extract_images_from_file_bytes(content, file.filename, output_dir="static/extracted_images")
            title = Path(file.filename).stem.replace('_', ' ').title()
            text = f"Graphic and visual document containing diagram or figure titled '{title}'."
            pipeline.add_document(text, file.filename, images=extracted_images)
            total_chunks = pipeline.n_chunks
            return {
                "message": f"Successfully ingested visual figure '{file.filename}' into Hopfield memory."
            }

        bio = io.BytesIO(content)
        text, _ = extract_text_from_file(bio, file.filename)
            
        if not text or not text.strip():
            raise ValueError(f"Extracted text from '{file.filename}' is empty or unreadable. Please check the document format.")
            
        print(f"[API] Upload request received: {file.filename} ({len(content)} bytes)", flush=True)
        extracted_images = extract_images_from_file_bytes(content, file.filename, output_dir="static/extracted_images")
        pipeline.add_document(text, file.filename, images=extracted_images)
        total_chunks = pipeline.n_chunks
        img_info = f", {len(extracted_images)} figures" if extracted_images else ""
        print(f"[API] Successfully ingested {file.filename}: {total_chunks} total chunks indexed ({len(extracted_images)} figures).", flush=True)
        return {
            "message": f"Successfully ingested {file.filename} ({total_chunks} total chunks{img_info} indexed in Hopfield memory).",
            "images_extracted": len(extracted_images)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# Answer Formats & Exam Rubrics Endpoints
# ==============================================================================

@app.get("/api/formats")
def list_formats():
    """List available answer formats and academic/technical mark rubrics."""
    from core.format_guidance import FORMAT_CATALOG
    catalog_list = []
    for fid, g in FORMAT_CATALOG.items():
        catalog_list.append({
            "format_id": fid,
            "name": getattr(g, "name", g.display_name),
            "display_name": g.display_name,
            "target_words": g.expected_word_range,
            "mark_weight": g.target_marks,
            "description": g.expected_word_range,
            "outline": g.structural_outline,
            "target_components": g.target_components
        })
    return {"formats": catalog_list}


# ==============================================================================
# Persistent Web Knowledge Memory Endpoints
# ==============================================================================

@app.get("/api/knowledge")
def list_knowledge():
    """List cached persistent web knowledge memory entries."""
    global pipeline
    if not pipeline or not hasattr(pipeline, "knowledge_memory"):
        return {"entries": [], "total_entries": 0}
    entries = pipeline.knowledge_memory.get_all_entries()
    return {
        "entries": [e.to_dict() for e in entries],
        "total_entries": len(entries)
    }


@app.delete("/api/knowledge")
def clear_knowledge():
    """Clear all persistent web knowledge memory entries."""
    global pipeline
    if not pipeline or not hasattr(pipeline, "knowledge_memory"):
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    pipeline.knowledge_memory.clear()
    return {"success": True, "message": "Persistent knowledge memory cleared"}


@app.delete("/api/knowledge/{entry_id}")
def delete_knowledge_entry(entry_id: str):
    """Delete a specific persistent knowledge memory entry."""
    global pipeline
    if not pipeline or not hasattr(pipeline, "knowledge_memory"):
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    found = pipeline.knowledge_memory.delete_entry(entry_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Knowledge entry '{entry_id}' not found")
    return {"success": True, "message": f"Knowledge entry '{entry_id}' deleted"}


@app.get("/")
async def get_index():
    response = FileResponse("static/index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

os.makedirs("static", exist_ok=True)
os.makedirs("static/extracted_images", exist_ok=True)
app.mount("/extracted_images", StaticFiles(directory="static/extracted_images"), name="extracted_images")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
