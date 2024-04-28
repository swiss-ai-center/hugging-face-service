import asyncio
import json
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import requests
from pydub import AudioSegment
import zipfile
import io

settings = get_settings()


class MyService(Service):
    """
    Hugging Face service uses Hugging Face's model hub API to directly query AI models
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Hugging Face",
            slug="hugging-face",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="json_description",
                    type=[
                        FieldDescriptionType.APPLICATION_JSON
                    ],
                ),
                FieldDescription(
                    name="hugging_input",
                    type=[
                        FieldDescriptionType.APPLICATION_ZIP
                    ]
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_ZIP]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.NATURAL_LANGUAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.NATURAL_LANGUAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        try:
            json_description = json.loads(data['json_description'].data.decode('utf-8'))
            api_token = json_description['api_token']
            api_url = json_description['api_url']
        except ValueError as err:
            raise Exception(f"json_description is invalid: {str(err)}")
        except KeyError as err:
            raise Exception(f"api_url or api_token missing from json_description: {str(err)}")
        headers = {"Authorization": f"Bearer {api_token}"}

        def extract_file_from_zip(zip_bytes):
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
                # there should be only one file in the zip
                file_name = zip_ref.namelist()[0]
                return zip_ref.read(file_name)

        def is_valid_json(json_string):
            try:
                json.loads(json_string)
                return True
            except ValueError:
                return False

        def create_zip_from_bytes(file_bytes, file_name):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                zip_file.writestr(file_name, file_bytes)
            zip_bytes = zip_buffer.getvalue()
            return zip_bytes

        def json_input_query(payload):
            response = requests.post(api_url, headers=headers, json=payload)
            return response

        def audio_or_image_input_query(file_bytes):
            response = requests.post(api_url, headers=headers, data=file_bytes)
            return response

        zip_file_bytes = data['hugging_input'].data
        file_data = extract_file_from_zip(zip_file_bytes)
        processed_data = None
        result_data = None

        if json_description['input_type'] == FieldDescriptionType.APPLICATION_JSON:
            json_payload = json.loads(file_data.decode('utf-8'))
            result_data = json_input_query(json_payload)
        elif json_description['input_type'] == FieldDescriptionType.AUDIO_MP3 or json_description['input_type'] == \
                FieldDescriptionType.IMAGE_PNG or FieldDescriptionType.IMAGE_JPEG == \
                json_description['input_type']:
            result_data = audio_or_image_input_query(file_data)

        if is_valid_json(result_data.content):
            json_data = json.loads(result_data.content)
            if 'error' in json_data:
                self._logger.error(json_data['error'])
                raise Exception(json_data['error'])

        match json_description['output_type']:
            case FieldDescriptionType.APPLICATION_JSON:
                result_data = json.dumps(result_data.json(), indent=4)
                processed_data = create_zip_from_bytes(result_data, "result.json")
            case FieldDescriptionType.IMAGE_PNG:
                processed_data = create_zip_from_bytes(result_data.content, "result.png")
            case FieldDescriptionType.IMAGE_JPEG:
                processed_data = create_zip_from_bytes(result_data.content, "result.jpg")
            case FieldDescriptionType.AUDIO_MP3:
                audio_segment = AudioSegment.from_file(io.BytesIO(result_data.content))
                processed_data = create_zip_from_bytes(audio_segment.export(format='mp3').read(), "result.mp3")
            case FieldDescriptionType.AUDIO_OGG:
                audio_segment = AudioSegment.from_file(io.BytesIO(result_data.content))
                processed_data = create_zip_from_bytes(audio_segment.export(format='ogg').read(), "result.ogg")

        return {
            "result": TaskData(data=processed_data,
                               type=FieldDescriptionType.APPLICATION_ZIP)
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """This service uses Hugging Face's model hub API to directly query AI models \n
You can choose from any model available on the inference API from the [Hugging Face Hub](https://huggingface.co/models)
 that takes image, audio or text(json) files as input and outputs one of the mentioned types.

This service has two input files:
 - A json file that defines the model you want to use, your access token and the input/output types you expect.
 - A zip file containing the input file.

json_description.json example:
 ```
 {
    "api_token": "your_token",
    "api_url": "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2",
    "input_type": "application/json",
    "output_type": "application/json"
}
```
This specific model "roberta-base-squad2" was trained on question-answer pairs, including unanswerable questions,
for the task of Question Answering.

The input looks like this:

```json
{
   "inputs":{
      "question":"What is my name?",
      "context":"My name is Clara Postlethwaite and I live in Berkeley."
   }
}
```
This is an example, check the model hub to see what the input of the model you want to use looks like.
Don't forget to compress it before giving it to the service!

The model may need some time to load on Hugging face's side, you may encounter an error on your first try.
Helpful trick: The answer from the inference API is cached, so if you encounter a loading error make sure to change the
input ever so slightly.
"""

api_summary = """A service that uses Hugging Face's model hub API to directly query AI models
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Hugging Face service",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
