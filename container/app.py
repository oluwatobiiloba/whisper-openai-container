import os
import json
import boto3
import whisper
import torch
import re
from typing import Dict, Any
from whisper.utils import get_writer
import sys

# Initialize S3 client
s3 = boto3.client("s3")

def generate_srt(transcription_result: Dict[str, Any]) -> str:
    """
    Generate an SRT subtitle file from transcription result.
    
    Args:
        transcription_result: Whisper transcription result containing the 'segments' (timing and text).
    
    Returns:
        str: Generated SRT content as a string.
    """
    srt_content = []
    for i, segment in enumerate(transcription_result["segments"]):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Convert time to SRT format (HH:MM:SS,SSS)
        start_time_str = format_time_to_srt(start_time)
        end_time_str = format_time_to_srt(end_time)

        srt_content.append(f"{i+1}")
        srt_content.append(f"{start_time_str} --> {end_time_str}")
        srt_content.append(f"{text}")
        srt_content.append("")  # Empty line after each segment

    return "\n".join(srt_content)

def format_time_to_srt(seconds: float) -> str:
    """
    Converts seconds into SRT time format (HH:MM:SS,SSS).
    
    Args:
        seconds: The time in seconds.
        
    Returns:
        str: The formatted time string for SRT.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def handler(event: Dict[str, str], context: Any) -> Dict[str, Any]:
    """
    Lambda handler to transcribe audio files, generate subtitles, and store results in S3.
    
    Args:
        event: Contains inputBucket, inputKey, outputKey
        context: Lambda context
    """
    try:
        # Extract S3 details
        bucket = event["inputBucket"]
        input_key = event["inputKey"]
        output_key = event["outputKey"]
        
        # Set up temp file paths
        os.makedirs("/tmp/data", exist_ok=True)
        audio_file = f"/tmp/data/{input_key}"
        
        # Download audio file from S3
        s3.download_file(bucket, input_key, audio_file)
        
        # Initialize Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("medium.en", download_root="/usr/local").to(device)
        
        # Perform transcription
        result = model.transcribe(audio_file, fp16=False, verbose=False, language="en")
        transcription = result["text"].strip()

        # Generate subtitles (SRT)
        srt_content = generate_srt(result)
        
        # Store transcription and subtitles in S3
        s3.put_object(
            Bucket=bucket,
            Key=f"{output_key}.text",
            Body=transcription
        )
        s3.put_object(
            Bucket=bucket,
            Key=f"{output_key}.srt",
            Body=srt_content
        )
        
        # Generate presigned URLs for both transcription and subtitle files
        try:
            text_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': f"{output_key}.text"},
                ExpiresIn=3600
            )
            srt_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': f"{output_key}.srt"},
                ExpiresIn=3600
            )
        except Exception as e:
            print(f"Error generating URLs: {e}")
            text_url = srt_url = "URL generation failed"
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "transcription": transcription,
                "s3_text_url": text_url,
                "s3_srt_url": srt_url
            })
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Error processing the file",
                "details": str(e)
            })
        }
    finally:
        # Cleanup temp files
        if os.path.exists(audio_file):
            os.remove(audio_file)