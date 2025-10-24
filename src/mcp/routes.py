"""
MCP Routes module for VX-RAG.

Defines API routes for MCP communication.
"""

from flask import Flask, request, jsonify
from .bridge import MCPBridge

app = Flask(__name__)

# TODO: Initialize bridge with query engine
# bridge = MCPBridge(query_engine)

@app.route("/query", methods=["POST"])
def query():
    """
    Endpoint for handling queries.
    """
    data = request.get_json()
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "Query is required"}), 400

    # response = bridge.handle_query(query_text)
    response = {"query": query_text, "response": "Placeholder response"}
    return jsonify(response)

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)