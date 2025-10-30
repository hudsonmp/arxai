#!/usr/bin/env python3
"""
WebSocket server for streaming annotation progress in real-time
"""
import os
import asyncio
import json
from typing import Dict, List
import websockets
from dotenv import load_dotenv

# Import annotation functions without modifying them
from arxiv_annotator import (
    get_highlight_suggestions,
    add_highlight_to_pdf,
    download_pdf_for_paper
)

try:
    import pymupdf as fitz
except ImportError:
    import fitz

load_dotenv(".env.local")

# Store connected clients
connected_clients = set()


async def send_to_clients(message: dict):
    """Send message to all connected WebSocket clients"""
    if connected_clients:
        message_json = json.dumps(message)
        await asyncio.gather(
            *[client.send(message_json) for client in connected_clients],
            return_exceptions=True
        )


async def stream_annotate_pdf(pdf_path: str, output_path: str, title: str, parent_title: str = "", parent_abstract: str = "", relevance_score: float = 0.0):
    """
    Annotate PDF with streaming progress updates, including comments and score header
    """
    await send_to_clients({
        "type": "start",
        "message": f"🎨 Starting annotation for: {title[:60]}...",
        "title": title,
        "relevance_score": relevance_score
    })
    
    try:
        doc = fitz.open(pdf_path)
        total_highlights = 0
        total_comments = 0
        max_pages = min(len(doc), 20)
        
        # Add relevance score header to first page
        if relevance_score > 0:
            first_page = doc[0]
            from arxiv_annotator import add_relevance_score_header
            add_relevance_score_header(first_page, relevance_score, parent_title)
            await send_to_clients({
                "type": "score_header",
                "message": f"✅ Added relevance score: {relevance_score:.4f}"
            })
        
        await send_to_clients({
            "type": "info",
            "message": f"📄 Processing {max_pages} pages..."
        })
        
        # Process each page
        for page_num in range(max_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if len(text.strip()) < 100:
                await send_to_clients({
                    "type": "skip",
                    "message": f"⏭️  Skipping page {page_num + 1} (too short)"
                })
                continue
            
            await send_to_clients({
                "type": "page_start",
                "message": f"📖 Processing page {page_num + 1}/{max_pages}...",
                "page": page_num + 1,
                "total_pages": max_pages
            })
            
            # Get highlight suggestions from Claude with parent context
            await send_to_clients({
                "type": "thinking",
                "message": "🤔 Asking Claude to identify key sections and relationships..."
            })
            
            highlights = await get_highlight_suggestions(page_num + 1, text, parent_title, parent_abstract)
            
            if not highlights:
                await send_to_clients({
                    "type": "no_highlights",
                    "message": f"   No highlights suggested for page {page_num + 1}"
                })
                continue
            
            await send_to_clients({
                "type": "highlights_found",
                "message": f"✨ Found {len(highlights)} potential highlights",
                "count": len(highlights)
            })
            
            # Add highlights with comments to PDF
            page_highlight_count = 0
            for h in highlights:
                text_snippet = h.get("text", "")
                reason = h.get("reason", "")
                priority = h.get("priority", 2)
                comment = h.get("relation_to_parent", "")
                
                priority_emoji = {1: "🔥", 2: "⭐", 3: "💡"}
                
                await send_to_clients({
                    "type": "highlight_attempt",
                    "message": f"{priority_emoji.get(priority, '•')} \"{text_snippet[:80]}...\"",
                    "text": text_snippet,
                    "reason": reason,
                    "priority": priority,
                    "relation": comment
                })
                
                if add_highlight_to_pdf(page, text_snippet, priority, comment):
                    page_highlight_count += 1
                    total_highlights += 1
                    if comment:
                        total_comments += 1
                    
                    await send_to_clients({
                        "type": "highlight_added",
                        "message": f"   ✅ Added with comment ({reason})",
                        "text": text_snippet,
                        "reason": reason,
                        "relation": comment
                    })
                else:
                    await send_to_clients({
                        "type": "highlight_failed",
                        "message": f"   ⚠️  Could not locate text in PDF"
                    })
            
            await send_to_clients({
                "type": "page_complete",
                "message": f"✅ Page {page_num + 1} complete: {page_highlight_count} highlights added",
                "page": page_num + 1,
                "highlights_added": page_highlight_count
            })
        
        # Save annotated PDF
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        
        await send_to_clients({
            "type": "complete",
            "message": f"🎉 Done! Added {total_highlights} highlights with {total_comments} comments",
            "total_highlights": total_highlights,
            "total_comments": total_comments,
            "output_path": output_path
        })
        
    except Exception as e:
        await send_to_clients({
            "type": "error",
            "message": f"❌ Error: {str(e)}"
        })


async def handle_client(websocket, path):
    """Handle WebSocket client connection"""
    connected_clients.add(websocket)
    print(f"✅ Client connected (total: {len(connected_clients)})")
    
    try:
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "🔌 Connected to annotation stream"
        }))
        
        async for message in websocket:
            data = json.loads(message)
            
            if data.get("action") == "start_annotation":
                # Use a test paper from the fallback directory
                pdf_path = data.get("pdf_path", "./fallback/10-1145-3617367_original.pdf")
                output_path = "./fallback/streamed_output.pdf"
                title = "Test Paper"
                
                await stream_annotate_pdf(pdf_path, output_path, title)
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        connected_clients.remove(websocket)


async def main():
    """Start WebSocket server"""
    print("\n" + "="*80)
    print("🚀 ArXiv Annotation Streaming Server")
    print("="*80)
    print(f"\n📡 WebSocket server starting on ws://localhost:8765")
    print(f"🌐 Open viewer at: http://localhost:8000/viewer.html")
    print(f"\nWaiting for connections...\n")
    
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())

