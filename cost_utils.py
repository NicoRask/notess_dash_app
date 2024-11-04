import pandas as pd

class LLMSystem:
    def __init__(self):
        # Define constants and default values
        # Defining average word count per content type
        self.words_per_page = 500  # according to GPT
        self.words_per_audio_min = 150  # according to GPT
        self.words_per_image_caption = 50  # Freely estimated
        self.words_per_image_face_recognition = 25  # Freely estimated
        self.words_per_image = self.words_per_image_caption + self.words_per_image_face_recognition

        # words per token according to OpenAI/GPT
        self.words_per_token = 0.75

        # tokens per content type
        self.tokens_per_page = self.words_per_page / self.words_per_token  # tokens per page = words per page / words per token
        self.tokens_per_audio_min = self.words_per_audio_min / self.words_per_token
        self.tokens_per_image = self.words_per_image / self.words_per_token

        # User-LLM conversation constants
        self.io_fraction = 0.1  # the fraction of input in an average conversation
        self.tokens_per_conversation = 2000  # average token usage
        self.questions_per_conversation = 5  # average questions per conversation with LLM
        self.output_conversations_fraction = 0.5  # the fraction of conversations requiring output in terms of blobs

        # Average sizes in kB
        self.avg_size_images = 2000  # Average size of images
        self.avg_size_page = 10  # Average size of a page
        self.avg_size_audio_per_minute = 5000  # Average size of audio per minute

        # Vector size in kB
        self.vector_size = 3  # rough estimate given by GPT

        # the chunks size in words
        # this could vary depending on what turns out to give good and stable results
        self.chunk_size_words = 200

        # Load Excel data
        inpath = "C:/Users/NicolaiRaskMathiesen/OneDrive - Rooftop Analytics ApS/Desktop/"
        fname = "costs.xlsx"
        self.LLM_cost = pd.read_excel(inpath + fname, sheet_name="LLM", index_col=0)
        self.embed_cost = pd.read_excel(inpath + fname, sheet_name="embedding", index_col=0)
        self.audio_cost = pd.read_excel(inpath + fname, sheet_name="audio", index_col=0)
        self.image_cap_cost = pd.read_excel(inpath + fname, sheet_name="image_captioning", index_col=0)
        self.image_face_rec_cost = pd.read_excel(inpath + fname, sheet_name="face_recognition", index_col=0)
        self.RAG_cost = pd.read_excel(inpath + fname, sheet_name="RAG", index_col=0)
        self.blob_cost = pd.read_excel(inpath + fname, sheet_name="blob", index_col=0)


    def get_media_to_text_costs(self, components, customer):
        # calculate cost for media to text conversion
        # here we assume that the straight forward cost listed by the service provider cover all non-negligeble costs
        cost_documents_to_text = 0 # for now assume trivial
        cost_audio_to_text = self.audio_cost.loc[components["audio"], "cost_per_min"] * customer.audio_min
        cost_image_cap = self.image_cap_cost.loc[components["image_captioning"], "cost_per_image"] * customer.images
        cost_face_recognition = (self.image_face_rec_cost.loc[components["face_recognition"], "cost_per_image"] *
                                 customer.images)
        cost_images_to_text = cost_image_cap + cost_face_recognition
        self.cost_media_to_text = cost_documents_to_text + cost_audio_to_text + cost_images_to_text
        return self.cost_media_to_text, cost_documents_to_text, cost_audio_to_text, cost_images_to_text


    def get_embedding_costs(self, components, customer):
        # get the cost associated with converting text to vector embeddings
        embed_cost = self.embed_cost
        cost_text = (embed_cost.loc[components["embedding"], "cost_per_token"] *
                     customer.pages * self.tokens_per_page)
        cost_audio = (embed_cost.loc[components["embedding"], "cost_per_token"] *
                      customer.audio_min * self.tokens_per_audio_min)
        cost_image = (embed_cost.loc[components["embedding"], "cost_per_token"] *
                      customer.images * self.tokens_per_image)
        self.cost_embedding = cost_text + cost_audio + cost_image
        return self.cost_embedding, cost_text, cost_audio, cost_image

    def get_RAG_costs(self, components, customer):
        # we approximate here that all text from documents is pooled and then divided into chunks
        n_vectors_doc = customer.pages * self.words_per_page / self.chunk_size_words
        # we approximate here that all text from audio is pooled and then divided into chunks
        n_vectors_audio = customer.audio_min * self.words_per_audio_min / self.chunk_size_words
        # we assume that each image text requires one vector
        n_vectors_image = customer.images
        # note that the actual number of vector could be a little higher depending on how small/big documents and
        # audio files are given by the user
        n_vectors = n_vectors_doc + n_vectors_audio + n_vectors_image
        storage_kb = n_vectors * self.vector_size
        cost_storage_per_month = storage_kb/1e6 * self.RAG_cost.loc[components["RAG"], "cost_storage_GB_month"]
        # we assume each item is written at upload time and thus separately. We assume writing is a one time thing for now.
        self.RAG_WU_cost = n_vectors * self.RAG_cost.loc[components["RAG"], "cost_per_WU"]
        # we assume a number of conversations with an average number of questions per conversation.
        # It is assumed that for each question the entire index of the RAG database is searched.
        # since the price is given per RU and one RU is equivalent to 1000 vectors being read we have to divide by 1000
        RU_cost_per_month = (n_vectors * self.RAG_cost.loc[components["RAG"], "cost_per_RU"] /
                             self.RAG_cost.loc[components["RAG"], "vectors_per_U"] * customer["conversations_per_month"] *
                             self.questions_per_conversation)
        self.total_RAG_cost_per_month = cost_storage_per_month + RU_cost_per_month
        return self.total_RAG_cost_per_month, cost_storage_per_month, storage_kb, self.RAG_WU_cost, RU_cost_per_month

    def get_blob_costs(self, components, customer):
        data_GB = (customer.pages * self.avg_size_page +
                   customer.audio_min * self.avg_size_audio_per_minute +
                   customer.images * self.avg_size_images) / 1e6
        # based on average size assumptions
        cost_storage_per_month = data_GB * self.blob_cost.loc[components["blob"], "cost_storage_GB_month"]
        # one write operation per file
        self.blob_cost_write = customer["files"] * self.blob_cost.loc[components["blob"], "cost_per_write_operation"]
        # it is assumed that a fraction of the conversations each month require retrieval of the original media.
        # it is also assumed that each time all files are read. This vastly overestimates the needed read operations.
        cost_read_per_month = (customer["files"] * self.blob_cost.loc[components["blob"], "cost_per_read_operation"] *
                     customer["conversations_per_month"] * self.questions_per_conversation *
                     self.output_conversations_fraction)
        self.total_blob_cost_per_month = cost_storage_per_month + cost_read_per_month
        return self.total_blob_cost_per_month, cost_storage_per_month, self.blob_cost_write, cost_read_per_month

    def get_LLM_costs(self, components, customer):
        # we assume a number conversations an average number of tokens per conversation (including context) and
        # a fraction of I/O in the conversations
        self.LLM_token_costs = (customer["conversations_per_month"] * self.tokens_per_conversation *
                           (self.LLM_cost.loc[components["LLM"], "cost_per_input_token"] * self.io_fraction +
                            self.LLM_cost.loc[components["LLM"], "cost_per_output_token"] * (1-self.io_fraction)))
        return self.LLM_token_costs

    def get_overall_costs(self, components, customer):
        self.get_media_to_text_costs(components, customer)
        self.get_embedding_costs(components, customer)
        self.get_RAG_costs(components, customer)
        self.get_blob_costs(components, customer)
        self.get_LLM_costs(components, customer)
        # LLM + RAG read + blob read + RAG storage + blob storage
        monthly_user_costs = self.LLM_token_costs + self.total_RAG_cost_per_month + self.total_blob_cost_per_month
        # media to text + embedding + RAG write + blob write
        data_creation_cost = self.cost_media_to_text + self.cost_embedding + self.RAG_WU_cost + self.blob_cost_write
        return monthly_user_costs, data_creation_cost