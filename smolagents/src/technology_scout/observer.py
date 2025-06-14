import os
import base64

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from dotenv import load_dotenv

load_dotenv()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_AUTH = base64.b64encode(
    f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
).decode()
LANGFUSE_HOST = "https://cloud.langfuse.com"
OTEL_EXPORTER_OTLP_ENDPOINT = "https://cloud.langfuse.com/api/public/otel"
OTEL_EXPORTER_OTLP_HEADERS = f"Authorization=Basic {LANGFUSE_AUTH}"

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = OTEL_EXPORTER_OTLP_ENDPOINT
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = OTEL_EXPORTER_OTLP_HEADERS

# Create a TracerProvider for OpenTelemetry
trace_provider = TracerProvider()

# Add a SimpleSpanProcessor with the OTLPSpanExporter to send traces
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# Instrument smolagents with the configured provider
SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


def setup_langfuse_tracer() -> None:
    """Setup the Langfuse tracer.

    This function is used to setup the Langfuse tracer.
    It does nothing.
    It is a reminder to import the module where the agent is built.
    """
