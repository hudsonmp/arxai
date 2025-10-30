#!/usr/bin/env python3
"""
Simple demo to show annotation streaming in terminal
"""
import asyncio
import websockets
import json

async def watch_stream():
    uri = "ws://localhost:8765"
    
    print("\n" + "="*80)
    print("📺 Watching Annotation Stream")
    print("="*80 + "\n")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to annotation server\n")
            
            # Send start command
            await websocket.send(json.dumps({"action": "start_annotation"}))
            print("🚀 Started annotation process...\n")
            print("-"*80 + "\n")
            
            # Receive and print messages
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type", "")
                msg_text = data.get("message", "")
                
                # Color-coded output
                if msg_type == "start":
                    print(f"\n🎬 {msg_text}\n")
                elif msg_type == "complete":
                    print(f"\n🎉 {msg_text}\n")
                    print("-"*80)
                    break
                elif msg_type == "page_start":
                    print(f"\n📄 {msg_text}")
                elif msg_type == "thinking":
                    print(f"   {msg_text}")
                elif msg_type == "highlights_found":
                    print(f"   {msg_text}")
                elif msg_type == "highlight_attempt":
                    print(f"      • {msg_text}")
                elif msg_type == "highlight_added":
                    print(f"        {msg_text}")
                elif msg_type == "page_complete":
                    print(f"   {msg_text}\n")
                elif msg_type == "error":
                    print(f"\n❌ {msg_text}\n")
                    break
                else:
                    print(f"   {msg_text}")
                    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(watch_stream())

