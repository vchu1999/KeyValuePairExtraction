import utils
import datetime

start_time = datetime.datetime.now()

utils.generate_documents()
end_time = datetime.datetime.now()
print(end_time - start_time)