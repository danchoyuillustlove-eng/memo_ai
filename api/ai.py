"""
AI プロンプト構築モジュール (AI Prompt Construction)

Notionへのデータ登録やチャット応答のために、AI（LLM）に送信するプロンプトを作成するロジックを担当します。
データベースのスキーマ構造や過去のデータ例をコンテキストとして組み込むことで、
AIが適切なJSON形式で回答できるように誘導します。
"""
import json
from typing import Dict, Any, List, Optional

from api.llm_client import generate_json, prepare_multimodal_prompt
from api.models import select_model_for_input


def construct_prompt(
    text: str,
    schema: Dict[str, Any],
    recent_examples: List[Dict[str, Any]],
    system_prompt: str
) -> str:
    """
    タスク抽出・プロパティ推定のための完全なプロンプトを構築します。
    
    Args:
        text (str): ユーザーの入力テキスト
        schema (Dict): 対象Notionデータベースのスキーマ情報
        recent_examples (List): データベースの直近の登録データ（Few-shot学習用）
        system_prompt (str): AIへの役割指示（システムプロンプト）
        
    Returns:
        str: LLMに送信するプロンプト文字列全体
    """
    # 1. スキーマ情報の整形
    # AIが理解しやすいように、Notionの複雑なスキーマオブジェクトを簡略化します。
    # 例: {"Status": "select options: ['未着手', '進行中', '完了']"}
    schema_info = {}
    for k, v in schema.items():
        schema_info[k] = v['type']
        # 選択肢があるタイプ（select, multi_select）の場合は、選択肢も列挙してAIに伝えます。
        if v['type'] == 'select':
            schema_info[k] += f" options: {[o['name'] for o in v['select']['options']]}"
        elif v['type'] == 'multi_select':
            schema_info[k] += f" options: {[o['name'] for o in v['multi_select']['options']]}"
            
    # 2. 過去データの整形 (Few-shot prompting)
    # 過去のデータ例を提示することで、AIに入力の傾向や期待するフォーマットを学習させます。
    examples_text = ""
    if recent_examples:
        for ex in recent_examples:
            props = ex.get("properties", {})
            simple_props = {}
            # プロパティの型に応じて値を抽出・簡略化
            for k, v in props.items():
                p_type = v.get("type")
                val = "N/A"
                if p_type == "title":
                    val = "".join([t.get("plain_text", "") for t in v.get("title", [])])
                elif p_type == "rich_text":
                    val = "".join([t.get("plain_text", "") for t in v.get("rich_text", [])])
                elif p_type == "select":
                    val = v.get("select", {}).get("name") if v.get("select") else None
                elif p_type == "multi_select":
                    val = [o.get("name") for o in v.get("multi_select", [])]
                elif p_type == "date":
                    val = v.get("date", {}).get("start") if v.get("date") else None
                elif p_type == "checkbox":
                    val = v.get("checkbox")
                simple_props[k] = val
            examples_text += f"- {json.dumps(simple_props, ensure_ascii=False)}\n"

    # プロンプトの組み立て
    # システムプロンプト + スキーマ定義 + データ例 + ユーザー入力 を結合
    prompt = f"""
{system_prompt}

Target Database Schema:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

Recent Examples:
{examples_text}

User Input:
{text}

Output JSON format strictly. NO markdown code blocks.
"""
    return prompt


def construct_chat_prompt(
    text: str,
    schema: Dict[str, Any],
    system_prompt: str,
    session_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    チャット対話用プロンプトの構築
    
    インタラクティブなチャット機能のためのプロンプトを作成します。
    会話履歴（session_history）を含めることで、文脈を踏まえた応答を可能にします。
    """
    schema_info = {}
    target_type = "database"
    
    # スキーマ判定: データベーススキーマかページスキーマか
    # データベースの場合は各プロパティの定義が含まれています。
    # ページの場合は固定スキーマ（Title, Contentなど）として扱います。
    
    for k, v in schema.items():
        if isinstance(v, dict) and "type" in v:
             schema_info[k] = v['type']
             if v['type'] == 'select' and 'select' in v:
                schema_info[k] += f" options: {[o['name'] for o in v['select']['options']]}"
             elif v['type'] == 'multi_select' and 'multi_select' in v:
                schema_info[k] += f" options: {[o['name'] for o in v['multi_select']['options']]}"
    
    # 会話履歴のテキスト化
    # 役割（Role）と内容（Content）を明記して、過去のやり取りを時系列で記述します。
    # Systemメッセージが含まれる場合は、AIへの追加指示として扱います（例：参照ページの本文など）。
    history_text = ""
    if session_history:
        for msg in session_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                 history_text += f"[System Info]: {content}\n"
            else:
                 history_text += f"{role.upper()}: {content}\n"

    # プロンプトテンプレートへの埋め込み
    # AIはJSON形式での応答を強制されます。
    # "properties" フィールドを埋めることで、会話の流れからタスク登録を行うことも可能です。
    prompt = f"""
{system_prompt}

Target Schema:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

Session History:
{history_text}

Current User Input:
{text if text else "(No text provided)"}

Restraints:
- You are a helpful AI assistant.
- Your output must be valid JSON ONLY.
- Structure:
{{
  "message": "Response to the user",
  "refined_text": "Refined version of the input, if applicable (or null)",
  "properties": {{ "Property Name": "Value" }} // Only if user intends to save data
}}
- If the user is just chatting, "properties" should be null.
- If the user wants to save/add data, fill "properties" according to the Schema.
"""
    return prompt


def validate_and_fix_json(json_str: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    AIのJSON応答を解析・検証・修正する関数
    
    LLMは時にMarkdownコードブロックを含んだり、不正なJSONを返したりするため、
    それらをクリーニングしてPython辞書として安全に取り出します。
    さらに、スキーマ定義に従って型変換（キャスト）を行い、Notion APIでエラーにならない形式に整えます。
    """
    # 1. Markdown記法の除去
    # ```json ... ``` のような装飾を取り除きます。
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # JSONパース失敗時の簡易リトライ
        # 余計な接頭辞/接尾辞がある場合に、最初の中括弧 { と最後の中括弧 } の間を抽出して再試行します。
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                data = json.loads(json_str[start:end])
            except Exception:
                # 復旧不能な場合は空の辞書を返して安全に終了
                return {}
        else:
             return {}

    # 2. プロパティの型検証とキャスト (Robust Property Validation)
    # Notion APIは型に厳格なため、スキーマ情報を基に各値を適切な形式に変換します。
    validated = {}
    for k, v in data.items():
        if k not in schema:
            continue
            
        target_type = schema[k]["type"]
        
        # 型ごとの詳細な処理
        if target_type == "select":
            # Select型: 文字列に変換
            if isinstance(v, dict): v = v.get("name")
            if v:
                validated[k] = {"select": {"name": str(v)}}
                
        elif target_type == "multi_select":
            # Multi-Select型: 文字列のリストに変換
            if not isinstance(v, list): v = [v]
            opts = []
            for item in v:
               if isinstance(item, dict): item = item.get("name")
               if item: opts.append({"name": str(item)})
            validated[k] = {"multi_select": opts}
            
        elif target_type == "status":
             # Status型
             if isinstance(v, dict): v = v.get("name")
             if v:
                 validated[k] = {"status": {"name": str(v)}}
                 
        elif target_type == "date":
            # Date型: YYYY-MM-DD 文字列を期待
            if isinstance(v, dict): v = v.get("start")
            if v:
                validated[k] = {"date": {"start": str(v)}}
                
        elif target_type == "checkbox":
            # Checkbox型: 真偽値
            validated[k] = {"checkbox": bool(v)}
            
        elif target_type == "number":
             # Number型: 数値変換
             try:
                 if v is not None:
                     validated[k] = {"number": float(v)}
             except (ValueError, TypeError):
                 # 数値変換に失敗した場合はスキップ
                 # 例: "abc" -> float() は ValueError
                 pass
                 
        elif target_type == "title":
             # Title型: Rich Text オブジェクト構造
             if isinstance(v, list): v = "".join([t.get("plain_text","") for t in v if "plain_text" in t])
             validated[k] = {"title": [{"text": {"content": str(v)}}]}
             
        elif target_type == "rich_text":
             # Rich Text型
             if isinstance(v, list): v = "".join([t.get("plain_text","") for t in v if "plain_text" in t])
             validated[k] = {"rich_text": [{"text": {"content": str(v)}}]}
             
        elif target_type == "people":
            # ユーザーIDが必要なため、現在は無視（実装難易度高）
            pass
            
        elif target_type == "files":
             # ファイルアップロードは複雑なため無視
             pass

    return validated


# --- NEW: High-level entry points ---

async def analyze_text_with_ai(
    text: str,
    schema: Dict[str, Any],
    recent_examples: List[Dict[str, Any]],
    system_prompt: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    テキスト分析とプロパティ抽出のメイン関数
    
    1. 最適なモデルの選択（テキストのみ/画像あり）
    2. プロンプトの構築
    3. LLMの呼び出し
    4. 結果の解析とプロパティのクリーニング
    を一括して行います。
    
    Args:
        text: ユーザー入力テキスト
        schema: Notionデータベーススキーマ
        recent_examples: 最近の登録データ（コンテキスト用）
        system_prompt: システムからの指示
        model: モデルの明示的な指定（省略時は自動選択）
    
    Returns:
        {
            "properties": {...},  # Notion登録用プロパティ
            "usage": {...},       # トークン使用量
            "cost": float,        # 推定コスト
            "model": str          # 使用されたモデル名
        }
    """
    # モデルの自動選択（この関数はテキスト入力のみを想定）
    selected_model = select_model_for_input(has_image=False, user_selection=model)
    
    # プロンプトの構築
    prompt = construct_prompt(text, schema, recent_examples, system_prompt)
    
    try:
        # LLM呼び出し
        result = await generate_json(prompt, model=selected_model)
        
        # プロパティの検証と修正
        properties = validate_and_fix_json(result["content"], schema)
        
        return {
            "properties": properties,
            "usage": result["usage"],
            "cost": result["cost"],
            "model": result["model"]
        }
    
    except Exception as e:
        print(f"AI Analysis Failed: {e}")
        
        # エラー時のフォールバック処理
        # AI分析に失敗しても、ユーザーの入力テキストをタイトルとして保存できるように
        # 最低限のプロパティ構造を作成して返します。
        fallback = {}
        for k, v in schema.items():
            if v["type"] == "title":
                fallback[k] = {"title": [{"text": {"content": text}}]}
                break
        
        return {
            "properties": fallback,
            "usage": {},
            "cost": 0.0,
            "model": selected_model,
            "error": str(e)
        }


async def chat_analyze_text_with_ai(
    text: str,
    schema: Dict[str, Any],
    system_prompt: str,
    session_history: Optional[List[Dict[str, str]]] = None,
    image_data: Optional[str] = None,
    image_mime_type: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    インタラクティブチャット分析のメイン関数 (画像対応)
    
    テキストだけでなく、画像データ（Base64）を含めたマルチモーダルな対話を処理します。
    会話履歴を考慮し、ユーザーとの自然な対話を行いながら、必要に応じてタスク情報（properties）を抽出します。
    
    Args:
        text: ユーザー入力テキスト
        schema: Notionの対象スキーマ
        system_prompt: システム指示
        session_history: 過去の会話履歴
        image_data: Base64エンコードされた画像データ（任意）
        image_mime_type: 画像のMIMEタイプ（任意）
        model: モデル指定
    
    Returns:
        dict: メッセージ、精製テキスト、抽出プロパティ、メタデータを含む辞書
    """
    # 画像の有無に基づくモデル自動選択
    has_image = bool(image_data and image_mime_type)
    print(f"[Chat AI] Has image: {has_image}, User model selection: {model}")
    selected_model = select_model_for_input(has_image=has_image, user_selection=model)
    print(f"[Chat AI] Selected model: {selected_model}")
    
    # 会話履歴の準備
    print(f"[Chat AI] Constructing messages, schema keys: {len(schema)}, history length: {len(session_history) if session_history else 0}")
    
    # スキーマ情報の整形
    schema_info = {}
    for k, v in schema.items():
        if isinstance(v, dict) and "type" in v:
             schema_info[k] = v['type']
             if v['type'] == 'select' and 'select' in v:
                schema_info[k] += f" options: {[o['name'] for o in v['select']['options']]}"
             elif v['type'] == 'multi_select' and 'multi_select' in v:
                schema_info[k] += f" options: {[o['name'] for o in v['multi_select']['options']]}"
    
    # システムプロンプトの構築
    system_message_content = f"""{system_prompt}

Target Schema:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

Restraints:
- You are a helpful AI assistant.
- Your output must be valid JSON ONLY.
- Structure:
{{
  "message": "Response to the user",
  "stamp": "😊",  // Optional: Use a single emoji stamp to express emotion (happy, thinking, surprised, etc.)
  "refined_text": "Refined version of the input, if applicable (or null)",
  "properties": {{ "Property Name": "Value" }} // Only if user intends to save data
}}
- If the user is just chatting, "properties" should be null.
- If the user wants to save/add data, fill "properties" according to the Schema.
- Use "stamp" field to express your emotion with a single emoji when appropriate (e.g., 😊 for happy, 🤔 for thinking, 😮 for surprised, 👍 for approval, ❤️ for appreciation). Only use when natural - not required for every response."""
    
    # メッセージ配列の構築
    messages = [{"role": "system", "content": system_message_content}]
    
    # 会話履歴を追加（画像データは含まれない、テキストのみ）
    if session_history:
        messages.extend(session_history)
    
    # 現在のユーザー入力を追加
    if has_image:
        #マルチモーダル: 画像データを含むコンテンツパーツを作成
        print(f"[Chat AI] Preparing multimodal message with image")
        current_user_content = prepare_multimodal_prompt(
            text or "(No text provided)",
            image_data,
            image_mime_type
        )
        messages.append({"role": "user", "content": current_user_content})
    else:
        # テキストのみ
        if text:
            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "user", "content": "(No text provided)"})
    
    # LLMの呼び出し（messages配列を渡す）
    print(f"[Chat AI] Calling LLM: {selected_model} with {len(messages)} messages")
    result = await generate_json(messages, model=selected_model)
    print(f"[Chat AI] LLM response received, length: {len(result['content'])}")
    json_resp = result["content"]
    
    # 応答データの解析
    try:
        data = json.loads(json_resp)
        
        # DEBUG: 生の解析結果をログ出力
        print(f"[Chat AI] Raw parsed response type: {type(data)}")
        print(f"[Chat AI] Raw parsed response: {data}")
        
        # 文字列が返ってきた場合の対応（LLMがJSON形式を返さなかった場合）
        if isinstance(data, str):
            print(f"[Chat AI] Response is a string, wrapping in message dict")
            data = {"message": data}
        
        # リスト形式で返ってきた場合の対応（一部のモデルの挙動）
        elif isinstance(data, list):
            print(f"[Chat AI] Response is a list, extracting first element")
            if data and isinstance(data[0], dict):
                data = data[0]
            else:
                data = {}

        if not data:
            data = {"message": "AIから有効な応答が得られませんでした。"}
            
        print(f"[Chat AI] After type handling: {data}")
        print(f"[Chat AI] Message field: {data.get('message') if isinstance(data, dict) else 'N/A'}")
        
    except json.JSONDecodeError:
        print(f"[Chat AI] JSON decode failed, attempting recovery from: {json_resp[:200]}")
        try:
            # 部分的なJSONの抽出によるリカバリ
            start = json_resp.find("{")
            end = json_resp.rfind("}") + 1
            data = json.loads(json_resp[start:end])
            print(f"[Chat AI] Recovered data: {data}")
        except Exception as e:
            print(f"[Chat AI] Recovery failed: {e}")
            data = {
                "message": "AIの応答を解析できませんでした。",
                "raw_response": json_resp
            }
    
    # フロントエンド向けのメッセージフィールド保証
    if "message" not in data or not data["message"]:
        print(f"[Chat AI] Message missing or empty, generating fallback")
        
        # プロパティが直接返された場合のフォールバックメッセージ生成
        has_properties = any(key in data for key in ["Title", "Content", "properties"])
        
        if "refined_text" in data and data["refined_text"]:
            data["message"] = f"タスク名を「{data['refined_text']}」に提案します。"
        elif has_properties:
            if "Title" in data or "Content" in data:
                title_val = data.get("Title", "")
                data["message"] = f"内容を整理しました: {title_val}" if title_val else "プロパティを抽出しました。"
            elif "properties" in data and data["properties"]:
                data["message"] = "プロパティを抽出しました。"
            else:
                data["message"] = "プロパティを抽出しました。"
        else:
            data["message"] = "（応答完了）"
        print(f"[Chat AI] Fallback message: {data['message']}")

    
    # データの正規化: AIがプロパティをトップレベルキーとして返した場合の修正
    if "properties" not in data:
        schema_keys = set(schema.keys())
        data_keys = set(data.keys())
        # メッセージ等以外のキーで、スキーマと一致するものをプロパティとみなす
        property_keys = data_keys.intersection(schema_keys)
        
        if property_keys:
            print(f"[Chat AI] Normalizing direct properties: {property_keys}")
            properties = {key: data[key] for key in property_keys}
            # トップレベルから削除して properties キー配下に移動
            for key in property_keys:
                del data[key]
            data["properties"] = properties
            print(f"[Chat AI] Normalized properties: {data['properties']}")
    
    # プロパティの詳細検証
    if "properties" in data and data["properties"]:
        data["properties"] = validate_and_fix_json(
            json.dumps(data["properties"]),
            schema
        )
    
    # メタデータの付与
    data["usage"] = result["usage"]
    data["cost"] = result["cost"]
    data["model"] = result["model"]
    
    print(f"[Chat AI] Final response data: {data}")
    
    return data

