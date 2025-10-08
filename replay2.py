import csv
import asyncio
import aiohttp
import json
from datetime import datetime
import ast
import argparse
from tqdm import tqdm  # Import tqdm for progress bar


def fire_request(session, method, url, headers, body, base_url, request_id):
    """Create a fire-and-forget request task"""
    headers_dict = ast.literal_eval(headers)
    # Remove headers that might cause issues
    headers_to_remove = ['content-length', 'host', 'connection']
    for header in headers_to_remove:
        headers_dict.pop(header, None)

    # Parse body if it's a string
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except:
            pass  # Keep original body if it can't be parsed as JSON
    
    # Set the model field in the body
    if isinstance(body, dict):
        body["model"] = "Sao10K/L3-8B-Lunaris-v1"
        # body["model"] = "meta-llama/Llama-3.1-70B-Instruct"
        
        # Debug: Let's see what we're sending for the first few requests
        if request_id < 5:
            print(f"Request {request_id} body keys: {list(body.keys())}")
            print(f"Request {request_id} model: {body.get('model')}")
            if 'prompt' in body:
                print(f"Request {request_id} prompt length: {len(body['prompt'])}")
            if 'max_tokens' in body:
                print(f"Request {request_id} max_tokens: {body['max_tokens']}")
    #     body = {
    #         "model": "Sao10K/L3-8B-Lunaris-v1",
    #         "prompt": "Once upon a time, there was a dragon named Fred...",
    #         "temperature": 0.7,
    #         "max_tokens": 500
    #     }

    # remove headers, use blank headers other than Content-Type: application/json
    headers_dict = {
        "Content-Type": "application/json"
    }

    # Return the coroutine without awaiting
    return session.post(
        url=base_url,
        headers=headers_dict,
        json=body if isinstance(body, dict) else body,
        ssl=False
        # No timeout - let the request complete naturally in background
    )

async def send_request_background(request_coro, request_id):
    """Handle request in background - fire and forget"""
    try:
        async with request_coro as response:
            # # Don't read response, just confirm it was sent
            # if request_id % 10 == 0:  # Print every 10th request
            print(f"Request {request_id} sent, got status: {response.status}")
            
            # For 404 errors, let's see the response content
            if response.status == 404:
                response_text = await response.text()
                print(f"Request {request_id} 404 response: {response_text[:200]}...")
            elif response.status == 200:
                # Track successful requests too
                if request_id < 10:
                    print(f"Request {request_id} SUCCESS")
            else:
                # Print any non-200, non-404 status
                response_text = await response.text()
                print(f"Request {request_id} status {response.status}: {response_text[:200]}...")
    except asyncio.TimeoutError:
        print(f"Request {request_id} failed: Timeout")
    except aiohttp.ClientError as e:
        print(f"Request {request_id} failed: Client error - {type(e).__name__}: {e}")
    except Exception as e:
        # Print all errors with more detail
        print(f"Request {request_id} failed: {type(e).__name__}: {e}")


async def replay_requests(csv_file, base_url, qps=6, max_request_num=None):
    # Disable timeout to allow requests to wait in queue as long as needed
    timeout = aiohttp.ClientTimeout(total=None)  # No timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)  # Read all rows into a list
            total_requests = len(rows)  # Get total number of requests
            
            # Limit the number of requests if specified
            if max_request_num is not None:
                rows = rows[:max_request_num]
                total_requests = len(rows)

            # Calculate interval between requests for target QPS
            interval = 1.0 / qps
            start_time = asyncio.get_event_loop().time()
            
            # Track all background tasks
            tasks = []

            for i, row in tqdm(enumerate(rows), total=total_requests, desc="Replaying requests"):
                # Calculate when this request should be sent
                target_time = start_time + (i * interval)
                current_time = asyncio.get_event_loop().time()
                
                # Sleep until it's time to send this request
                delay = target_time - current_time
                if delay > 0:
                    await asyncio.sleep(delay)

                # Fire request without waiting for completion
                request_coro = fire_request(
                    session,
                    row['method'],
                    base_url,
                    row['headers'],
                    row['body'],
                    base_url,
                    i
                )
                # Create background task that doesn't block QPS timing
                task = asyncio.create_task(send_request_background(request_coro, i))
                tasks.append(task)
                print(f"Fired request {i}")
            
            # Wait for all requests to complete
            print(f"\nAll {len(tasks)} requests sent. Waiting for responses...")
            await asyncio.gather(*tasks, return_exceptions=True)
            print("All requests completed.")


def start_replay(csv_file, base_url, qps=6, max_request_num=None):
    asyncio.run(replay_requests(csv_file, base_url, qps, max_request_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', required=True,
                        help='Path to the CSV file')
    parser.add_argument('--base-url', required=True,
                        help='Base URL for the API')
    parser.add_argument('--qps', type=float, default=6.0,
                        help='Queries per second (default: 6.0)')
    parser.add_argument('--max-request-num', type=int, default=None,
                        help='Maximum number of requests to send (default: all)')
    args = parser.parse_args()

    start_replay(args.csv_file, args.base_url, args.qps, args.max_request_num)

# exmaple command /data/conversation_qps_6.csv
# python lmcache-trace-analysis/replay2.py --csv-file conversation_id_10000_from_241227_qps_3_model_filtered_reordered.csv --base-url http://0.0.0.0:8020/v1/completions --qps 6.0 --max-request-num 10000
# python lmcache-trace-analysis/replay2.py --csv-file conversation_id_10000_from_241227_qps_3_model_filtered_reordered.csv --base-url http://0.0.0.0:8030/v1/completions --qps 6.0 --max-request-num 10000
# curl http://0.0.0.0:8020/metrics > lmcache_metrics_output.log
# curl http://0.0.0.0:8030/metrics > vllm_metrics_output.log
