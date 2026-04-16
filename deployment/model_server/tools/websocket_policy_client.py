# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging, argparse
import time, os
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client
import websockets.exceptions

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 300) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()
        
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")
            
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                    ping_interval=120,  # Increased from 20 to 120 seconds to handle long inference times
                    ping_timeout=120,   # Increased from 20 to 120 seconds to prevent premature connection closure
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
    
    @override
    def predict_action(self, query_info: Dict, max_retries: int = 3) -> Dict:
        """
        Predict action with automatic retry on connection failure.

        Args:
            query_info: Dictionary containing the query information
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Dictionary containing the prediction results

        Raises:
            RuntimeError: If the server returns an error message
            websockets.exceptions.ConnectionClosed: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                data = self._packer.pack(query_info)
                self._ws.send(data)
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")
                return msgpack_numpy.unpackb(response)
            except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
                logging.warning(
                    f"Connection closed during predict_action (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s, ...
                    backoff_time = 2 ** attempt
                    logging.info(f"Waiting {backoff_time}s before reconnecting...")
                    time.sleep(backoff_time)

                    logging.info(f"Attempting to reconnect to {self._uri}...")
                    try:
                        # Close old connection if it exists
                        try:
                            self._ws.close()
                        except Exception:
                            pass
                        # Reconnect to server
                        self._ws, self._server_metadata = self._wait_for_server()
                        logging.info(f"Successfully reconnected to {self._uri}")
                    except Exception as reconnect_error:
                        logging.error(f"Failed to reconnect: {reconnect_error}")
                        if attempt == max_retries - 2:
                            raise
                else:
                    logging.error(f"All {max_retries} retry attempts failed")
                    raise



