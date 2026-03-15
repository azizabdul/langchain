We need below 2 files under voices directory:
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json 

Run local_voice_agent.py If we face issues with jar version use command 1 else 2 would be enough

$env:KMP_DUPLICATE_LIB_OK = "TRUE";
streamlit run .\local_voice_agent.py
