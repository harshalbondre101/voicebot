from __future__ import annotations


def import_small_webrtc_transport():
    try:
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

        return SmallWebRTCTransport
    except Exception:
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        return SmallWebRTCTransport


def import_runner_types():
    from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments

    return RunnerArguments, SmallWebRTCRunnerArguments


def import_language_hi():
    try:
        from pipecat.transcriptions.language import Language

        return Language.HI
    except Exception:
        return None
