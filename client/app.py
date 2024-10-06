import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from MMAPIS.client.config.config import LOGGER_MODES,OPENAI_CONFIG,GENERAL_CONFIG,APPLICATION_PROMPTS,ALIGNMENT_CONFIG,TTS_CONFIG,USER_CONFIG
from MMAPIS.client.utils import *
import pandas as pd
from functools import partial
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth



if __name__ == "__main__":
    st.set_page_config(page_title="MMAPIS",
                       page_icon=":smiley:",
                       layout="wide",
                       initial_sidebar_state="auto")
    st.title("MMAPIS: Multi-Modal Academic Papers Interpretation System")
    authenticator = stauth.Authenticate(
        USER_CONFIG['credentials'],
        USER_CONFIG['cookie']['name'],
        USER_CONFIG['cookie']['key'],
        USER_CONFIG['cookie']['expiry_days'],
        USER_CONFIG['pre-authorized']
    )
    options = ["Register", "Login", "Reset Password"]
    option_dict = {option: index for index, option in enumerate(options)}
    register_option_placeholder = st.empty()
    option_menu_placeholder = st.empty()
    with option_menu_placeholder:
        register_option = option_menu("Options",
                                      options=options,
                                      default_index=option_dict.get(st.session_state.get("register_option"), 1), # default_index=1 if st.session_state.get("register_option", None) is None else option_dict[st.session_state.get("register_option")],
                                      icons=['house', 'gear', 'key'],
                                      menu_icon="cast",
                                      orientation="horizontal")

    st.session_state["register_option"] = register_option
    if st.session_state["register_option"] == "Register":
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
                pre_authorization=False,
                fields={'Form name': 'Register user',
                        'Email': 'Email',
                        'Username': 'Username(used to log in)',
                        'Password': 'Password',
                        'Repeat password': 'Repeat password',
                        'Register': 'Register'})
            if email_of_registered_user:
                save_config(config=USER_CONFIG)
                reset_register_option("Login")
                st.success('User registered successfully')

        except Exception as e:
            st.error(e)
    elif st.session_state["register_option"] == "Reset Password":
        try:
            username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password()
            if username_of_forgotten_password:
                USER_CONFIG["credentials"]["usernames"][username_of_forgotten_password]["password"] = new_random_password
                save_config(config=USER_CONFIG)
                st.success(
                    f"New password({new_random_password}) has been set for {username_of_forgotten_password}.")
                # The developer should securely transfer the new password to the user.
            elif username_of_forgotten_password == False:
                st.error('Username not found')
            else:
                st.warning('Please enter your username and email')
        except Exception as e:
            st.error(f"A problem occurred: {e}")

    else:
        authenticator.login(
            fields={'Form name': 'Login Interface',
                    'Username': 'your username',
                    'Password': 'your password',
                    'Login': 'Login'})
        foget_password_button = st.empty()
        t = foget_password_button.button("Forget Password",
                                         key="forget_password",
                                         on_click=reset_register_option,
                                         args=("Reset Password",))
        if st.session_state["authentication_status"] is False:
            st.error('Username or password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')
        elif st.session_state["authentication_status"]:
            option_menu_placeholder.empty()
            foget_password_button.empty()
            st.sidebar.write(f"Hello,{st.session_state.get('name', 'user')}. Welcome to MMAPIS")
            authenticator.logout(location="sidebar")
            #--------------------------------------init--------------------------------------
            user_id = st.session_state.get("name","user").replace(" ","_")
            ##--------------------------------------sidebar--------------------------------------##


            # display options

            var_api_key = st.sidebar.text_input("OpenAI API Key",
                                                None,
                                                key="api_key",
                                                help="Your OpenAI API key")
            var_api_key = var_api_key if var_api_key else OPENAI_CONFIG["api_key"]
            var_base_url = st.sidebar.text_input("OpenAI Base URL",
                                                 key="base_url",
                                                 value= None,
                                                 help="Your OpenAI base URL, e.g. https://api.openai.com/v1")
            var_base_url = var_base_url if var_base_url else OPENAI_CONFIG["base_url"]

            var_model = st.sidebar.selectbox("Model",
                                            MMAPIS_CLIENT_CONFIG.compatible_models,
                                            key="openai api model",
                                            help="OpenAI API model, e.g. gpt-3.5-turbo")

            with st.sidebar.expander("Display Options"):
                var_linelength = st.slider("Line Length",
                                           MMAPIS_CLIENT_CONFIG.line_length.min,
                                           MMAPIS_CLIENT_CONFIG.line_length.max,
                                           MMAPIS_CLIENT_CONFIG.line_length.default,
                                           help="Line length of arXiv search results in each line")
                var_max_links = st.slider("Max Links", MMAPIS_CLIENT_CONFIG.num_pdf.min,
                                                    MMAPIS_CLIENT_CONFIG.num_pdf.max,
                                                    MMAPIS_CLIENT_CONFIG.num_pdf.default,
                                                  help="Max links of arXiv search results")
                var_img_width = st.slider("Image Width",
                                          MMAPIS_CLIENT_CONFIG.img_width.min,
                                          MMAPIS_CLIENT_CONFIG.img_width.max,
                                          MMAPIS_CLIENT_CONFIG.img_width.default,
                                                  help="Image width of images in alignment results")

            with st.sidebar.expander("MMAPIS Configuration Options"):
                var_min_grained_level = st.slider("Min Grained Level",
                                                  MMAPIS_CLIENT_CONFIG.min_grained_level.min,
                                                  MMAPIS_CLIENT_CONFIG.min_grained_level.max,
                                                  MMAPIS_CLIENT_CONFIG.min_grained_level.default,
                                                  help="""Initial grid size for section summary; e.g., min_grained_level = 2 means the article is split based on subsections (`## Subsection`). 
                                                  Summaries will perform better with bigger min_grained_level, but it may take longer to process(since more fine-grained sections are generated)
                                                  and more API credits. 
                                                  """)
                var_max_grained_level = st.slider("Maximum Grid",
                                                  MMAPIS_CLIENT_CONFIG.max_grained_level.min,
                                                    MMAPIS_CLIENT_CONFIG.max_grained_level.max,
                                                    MMAPIS_CLIENT_CONFIG.max_grained_level.default,
                                         help="Maximum grid size for section summary")

                ignore_title_options = MMAPIS_CLIENT_CONFIG.ignore_titles
                ignore_title_map = MMAPIS_CLIENT_CONFIG.ignore_title_map
                # Define the multiselect widget
                var_ignore_titles = st.multiselect(
                    "Ignore Titles",
                    ignore_title_options,
                    default=MMAPIS_CLIENT_CONFIG.default_ignore_titles,
                    help="Titles to ignore in the processing, e.g., 'abstract', 'introduction', 'acknowledge'"
                )
                var_threshold = st.slider("Threshold",
                                          MMAPIS_CLIENT_CONFIG.threshold.min,
                                            MMAPIS_CLIENT_CONFIG.threshold.max,
                                            MMAPIS_CLIENT_CONFIG.threshold.default,
                                          help="Threshold for title-like keywords in similarity when aligning the document-level summary with the corresponding image")
                ignore_title_mapping = lambda x: ignore_title_map.get(x, x)
                var_ignore_titles = list(map(ignore_title_mapping, var_ignore_titles))

            # Model configuration options
            with st.sidebar.expander("OpenAI Model Configuration Options"):
                var_rpm_limit = st.slider("Openai API Rate Limitation",
                                                      MMAPIS_CLIENT_CONFIG.rpm_limit.min,
                                                    MMAPIS_CLIENT_CONFIG.rpm_limit.max,
                                                    MMAPIS_CLIENT_CONFIG.rpm_limit.default,
                                                      help="Number of processes for GPT model, if your API key is not limited, you can set it to 0, otherwise set it to 3 if you're limited by 3 requests per minute")
                max_tokens = get_max_tokens(MMAPIS_CLIENT_CONFIG.max_tokens_map, var_model)
                var_max_token = st.slider("Max Tokens",
                                          0, max_tokens, max_tokens,
                                          help="Max tokens for the GPT model")
                var_temperature = st.slider("Temperature",
                                            MMAPIS_CLIENT_CONFIG.temperature.min,
                                            MMAPIS_CLIENT_CONFIG.temperature.max,
                                            MMAPIS_CLIENT_CONFIG.temperature.default,
                                            help="Temperature of GPT model")
                var_top_p = st.slider("Top P",
                                      MMAPIS_CLIENT_CONFIG.top_p.min,
                                        MMAPIS_CLIENT_CONFIG.top_p.max,
                                        MMAPIS_CLIENT_CONFIG.top_p.default,
                                      help="Top P of GPT model")
                var_frequency_penalty = st.slider("Frequency Penalty",
                                                    MMAPIS_CLIENT_CONFIG.frequency_penalty.min,
                                                    MMAPIS_CLIENT_CONFIG.frequency_penalty.max,
                                                    MMAPIS_CLIENT_CONFIG.frequency_penalty.default,
                                                          help="Frequency penalty of GPT model")
                var_presence_penalty = st.slider("Presence Penalty",
                                                MMAPIS_CLIENT_CONFIG.presence_penalty.min,
                                                MMAPIS_CLIENT_CONFIG.presence_penalty.max,
                                                MMAPIS_CLIENT_CONFIG.presence_penalty.default,
                                                help="Presence penalty of GPT model")


            with st.sidebar.expander("Prompts Optional"):
                var_prompt_ratio = st.slider("Prompt Ratio",
                                            MMAPIS_CLIENT_CONFIG.prompt_ratio.min,
                                            MMAPIS_CLIENT_CONFIG.prompt_ratio.max,
                                            MMAPIS_CLIENT_CONFIG.prompt_ratio.default,
                                                     help="Prompt ratio of GPT model, e.g., 0.8 means up to 80% prompt and 20% response content")
                var_summary_prompt = st.file_uploader("Summary Prompt", type=["json"], key="summary_prompt",
                                                              help="Summary prompt JSON file, e.g., {'system': '', 'general_summary': ''}")
                var_integrate_prompt = st.file_uploader("Integrate Prompt", type=["json"],
                                                                key="integrate_prompt",
                                                                help="Integrate prompt JSON file, e.g., {'integrate_system': '', 'integrate': '', 'integrate_input': ''}")
                var_summary_prompt = load_json_file(var_summary_prompt)
                var_integrate_prompt = load_json_file(var_integrate_prompt)





            with st.sidebar.expander("Text-to-Speech Configuration Options"):
                var_tts_api_key = st.text_input("TTS API Key", key="tts_api_key", help="Your TTS API key")
                var_tts_base_url = st.text_input("TTS Base URL", key="tts_base_url", help="Your TTS base URL")
                var_app_secret = st.text_input("App Secret", key="app_secret", help="Your app secret")



            #--------------------------------------main--------------------------------------

            args = Args()
            selection = option_menu(
                None,
                ["Search with URL", "Upload PDF"],
                icons=['house', 'cloud-upload'],
                menu_icon="cast",
                default_index=1,
                orientation="horizontal"
            )
            # Check for OpenAI API key
            if not var_api_key:
                st.error("Please input your OpenAI API key.")
                st.stop()
            # Search with URL logic
            if selection == "Search with URL":
                st.subheader("Choose search options:")
                var_daily_type = st.selectbox(
                    'Daily Subject Type',
                    ARXIV_REQUEST_CONFIG.daily_type,
                    help="Daily type of arXiv search, when keyword is None, this option will be used."
                )

                col1, col2, col3, col4 = st.columns(4)
                var_show_abstracts = col1.selectbox(
                    "Show Abstracts",
                    ARXIV_REQUEST_CONFIG.show_abstracts,
                    key="show_abstracts",
                    help="Show abstract of arXiv search result."
                )
                var_searchtype = col2.selectbox(
                    "Choose Search Type",
                    ARXIV_REQUEST_CONFIG.searchtype,
                    key="searchtype",
                    help=("Search type of arXiv search. "
                          "Options include: All, Title, Abstract, Author, Comment, "
                          "Journal Ref, Subject Class, Report Num, ID List.")
                )
                var_order = col3.selectbox(
                    "Choose Search Order",
                    ARXIV_REQUEST_CONFIG.search_order,
                    key="order",
                    help=("Order of arXiv search result. "
                          "-announced_date_first: Prioritizes by descending order of the earliest announcement date. "
                          "submitted_date: Sorts in ascending order based on submission date. "
                          "-submitted_date: Organizes in descending order by submission date. "
                          "announced_date_first: Arranges by ascending order of the earliest announcement date.")
                )
                var_size = col4.selectbox(
                    "Search Size",
                    ARXIV_REQUEST_CONFIG.search_size,
                    key="size",
                    help="Size of arXiv search result."
                )
                # Keyword input
                st.text("Input keyword to search specific fields in arXiv papers.")
                var_keyword = st.text_input(
                    "Keyword of arXiv Search",
                    "None",
                    key="keyword",
                    help="Keyword of arXiv search, e.g., 'quantum computer'."
                )

                result_placeholder = st.empty()
                if var_keyword.lower() == "none":
                    var_keyword = None
                    result_placeholder.text("No keyword input, automatically recommending papers delivered today.")
                else:
                    result_placeholder.text(f"Searching '{var_keyword}'...")
                var_links,var_titles,var_abstract,var_authors = get_links(var_keyword,
                                                                          max_num=var_max_links,
                                                                          line_length=var_linelength,
                                                                          searchtype=var_searchtype,
                                                                          order=var_order,
                                                                          size=var_size,
                                                                          daily_type=var_daily_type,
                                                                          markdown=True,
                                                                          user_name=user_id,
                                                                          )

                if var_links is None:
                    result_placeholder.text(f"Sorry, search failed. Error message: {var_authors}. Please retry.")
                    st.stop()

                var_options = [(i,False) for i in range(len(var_links))]
                # Display search results
                if var_keyword == "None":
                    result_placeholder.text(
                        f"Total {var_max_links} papers delivered today. Please choose the links you want to parse (you can choose multiple links):")
                else:
                    result_placeholder.text(
                        f"Searched '{var_keyword}' successfully. Found {var_max_links} related contents. Links as follows (you can choose multiple links):")

                # Choice list for checkboxes and abstracts
                choice_list = []
                for i,link in enumerate(var_links):
                    selected = st.checkbox(f"{var_titles[i]}", key=f"link{i + 1}",value=False)
                    var_options[i] = (i,selected)
                    abs_text = st.empty()
                    abs_text.markdown(f"{var_authors[i]}<br><br>{var_abstract[i]}",unsafe_allow_html=True)
                    choice_list.append([selected,abs_text])

                # Divider
                st.divider()

                # Selected record
                selected_record = st.empty()
                var_num_selected = sum(selected for _, selected in var_options)
                selected_record.text(f"You have selected {var_num_selected} links.")
                run_button = st.button(
                    "Run Model",
                    key="run_model_url",
                    help="Run model to generate summaries."
                )

                st.divider()

                # Display selected links
                st.text("You have selected the following links:")
                var_selected_links = filter_links(var_options)

                # Session state management
                var_variables = {key: value for key, value in globals().items() if key.startswith("var_") and not callable(value)}
                init_session_state()

                check_session_state(var_variables)

                if run_button or st.session_state["run_model"]:
                    if var_num_selected == 0:
                        st.error("You haven't selected any links. Please select links before running the model.")
                        st.stop()

                    # Initialize session state if needed
                    if run_button:
                        init_session_state(url_reset=True)

                    progress_text = 'Processing...'
                    progress_bar = st.progress(0, text=progress_text)

                    # Clear previous selections if necessary
                    if st.session_state["run_model"]:
                        choice_list = [[x[0],x[1].empty()] for x in choice_list if not x[0]]

                    time_record = []
                    pdf_record = []
                    length_record = []
                    # Processing each selected PDF
                    progress_step = ["Nougat Model Prediction", "Document-Level Summary Generation"]

                    for i,index in enumerate(var_selected_links):
                        progress_text = f"Processing PDF {i + 1}/{var_num_selected}: {var_titles[index]}"
                        progress_bar.progress((i) / var_num_selected, text=progress_text)
                        progress_time = []
                        args.pdf = [var_links[index]]
                        user_input = st.chat_message("user")
                        user_input.caption(f"Input Pdf :")
                        user_input.markdown(f"{var_titles[index]}")

                        now_time = time.time()
                        answer_message = st.chat_message("assistant")

                        # Nougat Model Prediction
                        pdf_page_num = get_pdf_length(args.pdf[0])
                        file_id = generate_cache_key(user_id=user_id, **vars(args))
                        model_result,pdf_name = run_model_with_progress(
                            estimated_time=estimate_model_time(current_stage=1,input_size=pdf_page_num,model=var_model),
                            current_stage=1,
                            _stage_function=get_model_predcit,
                            total_stages=2,
                            pdf_content=None,
                            user_name=user_id,
                            file_id=file_id,
                            **vars(args)
                        )

                        elapsed_time = time.time() - now_time
                        progress_time.append(elapsed_time)

                        if not model_result:
                            if pdf_name:
                                st.error(f"Model run failed. Error message: {pdf_name}. Please retry.")
                            else:
                                st.error("Model run failed. Error message: No content returned. Please try again or delete the file if you have already tried it.")
                            continue
                        else:
                            model_result, pdf_name = model_result[0], pdf_name[0]
                            num_sections = estimate_num_sections(model_result,min_grained_level=var_min_grained_level)
                            req_params = {
                                "api_key": var_api_key,
                                "base_url": var_base_url,
                                "min_grained_level": var_min_grained_level,
                                "max_grained_level": var_max_grained_level,
                                "summary_prompts": var_summary_prompt,
                                "document_prompts": var_integrate_prompt,
                                "pdf": args.pdf[0],
                                "img_width": var_img_width,
                                "threshold": var_threshold,
                                "summarizer_params": {
                                    "rpm_limit": var_rpm_limit,
                                    "ignore_titles": var_ignore_titles,
                                    "prompt_ratio": var_prompt_ratio,
                                    "gpt_model_params": {
                                        "model": var_model,
                                        "max_tokens": var_max_token,
                                        "temperature": var_temperature,
                                        "top_p": var_top_p,
                                        "frequency_penalty": var_frequency_penalty,
                                        "presence_penalty": var_presence_penalty,
                                    }
                                },
                            }

                            request_id = generate_cache_key(user_id=user_id,**req_params)

                            document_level_summary, section_level_summary, document_level_summary_aligned = run_model_with_progress(
                                estimated_time=estimate_model_time(current_stage=2, input_size=num_sections,model=var_model),
                                current_stage=2,
                                _stage_function=get_document_level_summary,
                                total_stages=2,
                                user_name=user_id,
                                file_id=file_id,
                                request_id=request_id,
                                from_middleware=False,
                                raw_md_text=model_result,
                                file_name=pdf_name,
                                **req_params
                            )


                            elapsed_time = time.time() - now_time
                            progress_time.append(elapsed_time - progress_time[-1])

                            # Display processing times
                            df = pd.DataFrame([progress_time], columns=progress_step,index=["Used Time"])
                            answer_message.caption(f"Parser Result:")
                            answer_message.dataframe(df,use_container_width=True,column_config=
                            {
                                "Nougat Model Prediction": st.column_config.NumberColumn(
                                    "Used Time of Model Prediction", format="%.2f s"),
                                "Document-Level Summary Generation": st.column_config.NumberColumn(
                                    "Used Time of Summary Generation", format="%.2f s"),
                            })

                            if not document_level_summary:
                                st.error(f"Summary generation failed, error message:{document_level_summary_aligned},please retry")
                                continue

                            if run_button:
                                st.session_state["generated_summary"][i] = [document_level_summary_aligned]
                            tabs = [f":{MMAPIS_CLIENT_CONFIG.color_ls[k]}[Generation  {k}]" for k in range(st.session_state["num_pages"][i]+1)]

                            for page_idx,tab_idx in enumerate(answer_message.tabs(tabs)):
                                tab_idx.markdown(st.session_state["generated_summary"][i][page_idx],unsafe_allow_html=True)

                                enhance_answer = partial(get_enhance_answer,
                                                         api_key=var_api_key,
                                                         base_url=var_base_url,
                                                         section_level_summary=section_level_summary,
                                                         index=i,
                                                         page_idx=page_idx,
                                                         raw_md_text=model_result,
                                                         pdf=args.pdf[0],
                                                         min_grained_level=var_min_grained_level,
                                                         max_grained_level=var_max_grained_level,
                                                         threshold=var_threshold,
                                                         img_width=var_img_width,
                                                         summarizer_params={
                                                             "rpm_limit": OPENAI_CONFIG["rpm_limit"],
                                                             "ignore_titles": var_ignore_titles,
                                                             "num_processes": var_rpm_limit,
                                                             "prompt_ratio": var_prompt_ratio,
                                                             "gpt_model_params":
                                                                 {
                                                                     "model": var_model,
                                                                     "max_tokens": var_max_token,
                                                                     "temperature": var_temperature,
                                                                     "top_p": var_top_p,
                                                                     "frequency_penalty": var_frequency_penalty,
                                                                     "presence_penalty": var_presence_penalty,
                                                                 }
                                                         },
                                                         user_name=user_id,
                                                         file_id= pdf_name,
                                                         request_id=request_id,
                                                        )

                                rerun_col, blog_col, speech_col,recommend_col,qa_col = tab_idx.columns(5)
                                rerun_button = rerun_col.button(":repeat: regenerate",
                                                                on_click=enhance_answer,
                                                                kwargs={"usage": 'regenerate',
                                                                        "prompts": APPLICATION_PROMPTS["regenerate_prompts"],
                                                                        "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                                                                        },
                                                                help="regenerate summary",
                                                                key=f"rerun_{i}_{page_idx}")
                                blog_button = blog_col.button(":memo: blog",
                                                              on_click=enhance_answer,
                                                              kwargs={"usage": 'blog',
                                                                      "prompts": APPLICATION_PROMPTS["blog_prompts"],
                                                                      "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                                                                      },
                                                              help="blog summary",
                                                              key=f"blog_{i}_{page_idx}")
                                speech_button = speech_col.button(":loud_sound: speech",
                                                                  on_click = enhance_answer,
                                                                  kwargs={"usage": 'speech',
                                                                          "prompts": APPLICATION_PROMPTS["broadcast_prompts"],
                                                                          "tts_api_key": TTS_CONFIG['api_key'],
                                                                            "tts_base_url": TTS_CONFIG['base_url'],
                                                                            "app_secret": TTS_CONFIG['app_secret'],
                                                                            "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                                                                            },
                                                                  help="speech summary",
                                                                  key=f"speech_{i}_{page_idx}")
                                recommend_button = recommend_col.button(":loud_sound: recommend",
                                                                    on_click = enhance_answer,
                                                                    kwargs={"usage": 'recommend',
                                                                            "prompts": APPLICATION_PROMPTS["score_prompts"],
                                                                            "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                                },
                                                                    help="recommend summary",
                                                                    key=f"recommend_{i}_{page_idx}")
                                qa_button = qa_col.button(":question: qa",
                                                            on_click = enhance_answer,
                                                            kwargs={"usage": 'qa',
                                                                    "prompts": APPLICATION_PROMPTS["multimodal_qa"],
                                                                    "document_level_summary": st.session_state["generated_summary"][i][page_idx]
                                },
                                                            help="qa summary",
                                                            key=f"qa_{i}_{page_idx}")

                                dur = int(time.time() - now_time)
                            time_record.append(progress_time)
                            pdf_record.append(var_links[index])
                            length_record.append(pdf_page_num)
                            st.divider()
                        time_df = pd.DataFrame(time_record,columns=progress_step)
                    progress_bar.progress(var_num_selected / var_num_selected, text=f"{var_num_selected} PDF Completed")
                    st.subheader("Running Result:")
                    if var_num_selected == 0:
                        st.warning("No links selected. Please select links before viewing the running result.")
                    else:
                        st.text(f"You have selected {var_num_selected} links, the running time is as follows:")
                        time_df.insert(0, "pdf_length", length_record)
                        time_df.insert(0, "pdf_link", pdf_record)
                        st.text(f"Running time details:")
                        st.dataframe(time_df, column_config={
                            "pdf_link": st.column_config.LinkColumn("PDF Link"),
                            "pdf_length": st.column_config.NumberColumn("Number of Pages", format="%d pages"),
                            **{step: st.column_config.NumberColumn(step, format="%.2f s") for step in progress_step}
                        }, hide_index=True, use_container_width=True)

            else:
                uploaded_file = st.file_uploader(label='Upload Local PDF Files (PDF/ZIP)',
                                                 accept_multiple_files=True,
                                                type=['pdf', 'zip'], key="upload_file",help="Upload one or more PDF or ZIP files.")
                progress_step = ["Nougat Model Prediction", "Document-Level Summary Generation"]
                if uploaded_file:
                    # pdf_content_list:List[Bytes],var_file_names:List[str]
                    pdf_content_list, var_file_names = load_file(uploaded_file)
                    # remove duplicate pdf
                    pdf_info_df = pd.DataFrame({"pdf_name": var_file_names, "pdf_content": pdf_content_list})
                    pdf_info_df = pdf_info_df.drop_duplicates(subset=["pdf_name"])
                    pdf_content_list, var_file_names = pdf_info_df["pdf_content"].tolist(), pdf_info_df["pdf_name"].tolist()
                    flag, pdf_content_list = upload_pdf_sync_cache(
                        pdf_bytes=pdf_content_list,
                        user_name=user_id,
                        file_names=var_file_names,
                        file_ids=var_file_names,
                        temp_file= False,
                        file_type="pdf"
                    )
                    if not flag:
                        st.error(f"Upload PDF to API failed. Due to {pdf_content_list}")
                        st.stop()
                    duplicate_pdf_count = len(uploaded_file) - len(pdf_content_list)
                    var_pdf_list = pdf_content_list

                    if duplicate_pdf_count > 0:
                        st.text(f"Found {duplicate_pdf_count} duplicate PDF files, which were automatically removed.")
                    if len(var_file_names) == 0 or any([pdf is None for pdf in var_file_names]):
                        st.error("Please upload valid PDF/ZIP files.")

                    pdf_button = st.button("Run Model",
                                           key="upload_run_model",
                                           help="Run the model to generate summaries")


                    #--------------------------------------session state--------------------------------------
                    var_variables = {key: value for key, value in globals().items() if
                                     key.startswith("var_") and not callable(value)}

                    # Initialize session state and handle updates.
                    init_session_state()
                    check_session_state(var_variables,upload_pdf=True)
                    # Process PDFs when the button is clicked or when session state indicates a change.
                    if pdf_button or st.session_state["pdf_run_model"]:
                        if pdf_button:
                            init_session_state(pdf_reset=True)

                        num_selected = len(var_file_names)
                        st.write(f"You have uploaded {num_selected} PDF files: {var_file_names}")

                        # Initialize progress bar with a clear description.
                        progress_text = "Processing..."
                        process_bar = st.progress(0, text=progress_text)

                        # Process each PDF file.
                        for i, pdf, name in zip(range(num_selected), pdf_content_list, var_file_names):
                            progress_text = f"Processing PDF {i + 1}/{num_selected}: {name}"
                            process_bar.progress((i) / num_selected, text=progress_text)

                            # Prepare arguments and display information about the input PDF.
                            progress_time_list = []
                            args.pdf_name = name
                            user_input = st.chat_message("user")
                            user_input.caption(f"input pdf :")
                            user_input.markdown(f"{name}")
                            user_input.download_button(label=f"Download {name}", data=pdf,
                                                       file_name=f"{str(name)}", key=f"download_{str(name)}")

                            # Start processing the PDF.
                            response_msg = st.chat_message("assistant")
                            st_time = time.time()

                            # Get number of pages and run the model.
                            num_pages = get_pdf_length(pdf)
                            args.pdf = [pdf]
                            file_id = generate_cache_key(user_id=user_id, **vars(args))
                            model_result, pdf_name = run_model_with_progress(
                                estimated_time=estimate_model_time(current_stage=1, input_size=num_pages,model=var_model),
                                current_stage=1,
                                _stage_function=get_model_predcit,
                                total_stages=2,
                                pdf_content=None,
                                user_name=user_id,
                                file_id=file_id,
                                **vars(args)
                            )
                            elapsed_time = time.time() - st_time
                            progress_time_list.append(elapsed_time)

                            # Handle errors during model execution.
                            if model_result is None:
                                st.error(f"Processing {name} failed, error message: {pdf_name}, please retry")
                                continue

                            # Extract results from the tuple.
                            model_result, pdf_name = model_result[0], pdf_name[0]

                            pdf_req_params = {
                                "api_key": var_api_key,
                                "base_url": var_base_url,
                                "min_grained_level": var_min_grained_level,
                                "max_grained_level": var_max_grained_level,
                                "summary_prompts": var_summary_prompt,
                                "document_prompts": var_integrate_prompt,
                                "pdf": args.pdf[0],
                                "img_width": var_img_width,
                                "threshold": var_threshold,
                                "summarizer_params": {
                                    "rpm_limit": var_rpm_limit,
                                    "ignore_titles": var_ignore_titles,
                                    "prompt_ratio": var_prompt_ratio,
                                    "gpt_model_params": {
                                        "model": var_model,
                                        "max_tokens": var_max_token,
                                        "temperature": var_temperature,
                                        "top_p": var_top_p,
                                        "frequency_penalty": var_frequency_penalty,
                                        "presence_penalty": var_presence_penalty,
                                    }
                                },
                            }

                            pdf_request_id = generate_cache_key(user_id=user_id,
                                                                **pdf_req_params)

                            # Generate summaries.
                            document_level_summary, section_level_summary, document_level_summary_aligned  = run_model_with_progress(
                                estimated_time=estimate_model_time(current_stage=2,
                                                                   input_size=estimate_num_sections(model_result,min_grained_level=var_min_grained_level),
                                                                   model=var_model),
                                current_stage=2,
                                _stage_function=get_document_level_summary,
                                total_stages=2,
                                user_name=user_id,
                                file_id=file_id,
                                request_id=pdf_request_id,
                                from_middleware=True,
                                raw_md_text= model_result,
                                file_name=pdf_name,
                                **pdf_req_params
                            )
                            # Display the processing times.
                            elapsed_time = time.time() - st_time
                            progress_time_list.append(elapsed_time - progress_time_list[-1])
                            df = pd.DataFrame([progress_time_list], columns=progress_step, index=["Used Time"])

                            response_msg.caption(f"parser result:")
                            response_msg.dataframe(df, use_container_width=True, column_config=
                            {
                                "Nougat Model Prediction": st.column_config.NumberColumn(
                                    "Used Time of Model Prediction", format="%.2f s",
                                ),
                                "Document-Level Summary Generation": st.column_config.NumberColumn(
                                    "Used Time of Summary Generation", format="%.2f s",
                                ),
                            })

                            # Handle errors during summary generation.
                            if document_level_summary is None:
                                response_msg.error(f"Abstract generation failed, error message:{document_level_summary_aligned},please retry")
                                continue

                            # Store the generated summary.
                            if pdf_button:
                                st.session_state["pdf_generated_summary"][i] = [document_level_summary_aligned]
                            pdf_tabs = [f":{MMAPIS_CLIENT_CONFIG.color_ls[k]}[Generation {k}]" for k in range(st.session_state["pdf_num_pages"][i] + 1)]

                            for pdf_page_idx, pdf_tab_idx in enumerate(response_msg.tabs(pdf_tabs)):
                                pdf_tab_idx.markdown(st.session_state["pdf_generated_summary"][i][pdf_page_idx],unsafe_allow_html=True)

                                # Function for enhancing answers (partial application).
                                enhance_answer = partial(get_enhance_answer,
                                                            api_key=var_api_key,
                                                            base_url=var_base_url,
                                                            section_level_summary=section_level_summary,
                                                            index=i,
                                                            pdf=pdf,
                                                            page_idx=pdf_page_idx,
                                                            raw_md_text=model_result,
                                                            min_grained_level=var_min_grained_level,
                                                            max_grained_level=var_max_grained_level,
                                                            threshold=var_threshold,
                                                            url_mode = False,
                                                            img_width=var_img_width,
                                                            summarizer_params={
                                                                "rpm_limit": var_rpm_limit,
                                                                "ignore_titles": var_ignore_titles,
                                                                "prompt_ratio": var_prompt_ratio,
                                                                "gpt_model_params":
                                                                    {
                                                                        "model": var_model,
                                                                        "max_tokens": var_max_token,
                                                                        "temperature": var_temperature,
                                                                        "top_p": var_top_p,
                                                                        "frequency_penalty": var_frequency_penalty,
                                                                        "presence_penalty": var_presence_penalty,
                                                                    }
                                                            },
                                                            user_name=user_id,
                                                            file_id=pdf_name,
                                                            request_id=pdf_request_id,
                                                            )
                                rerun_col, blog_col, speech_col, recommend_col,qa_col = pdf_tab_idx.columns(5)
                                rerun_button = rerun_col.button(":arrows_counterclockwise: regenerate",
                                                                on_click=enhance_answer,
                                                                kwargs={
                                                                    "usage": 'regenerate',
                                                                    "prompts": APPLICATION_PROMPTS["regenerate_prompts"],
                                                                    "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                                },
                                                                help="regenerate summary",
                                                                key=f"rerun_{i}_pdf_{pdf_page_idx}")
                                blog_button = blog_col.button(":memo: blog", key=f"blog_{i}_pdf_{pdf_page_idx}",
                                                                on_click=enhance_answer,
                                                                kwargs={
                                                                    "usage": 'blog',
                                                                    "prompts": APPLICATION_PROMPTS["blog_prompts"],
                                                                    "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                                },
                                                                help="blog summary")
                                speech_button = speech_col.button(":loud_sound: speech",
                                                                  key=f"speech_{i}_pdf_{pdf_page_idx}",
                                                                  on_click=enhance_answer,
                                                                  kwargs={"usage": 'speech',
                                                                            "prompts": APPLICATION_PROMPTS["broadcast_prompts"],
                                                                          "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                                          },
                                                                  help="speech summary")
                                recommend_button = recommend_col.button(":loud_sound: recommend",
                                                                    key=f"recommend_{i}_pdf_{pdf_page_idx}",
                                                                    on_click=enhance_answer,
                                                                    kwargs={"usage": 'recommend',
                                                                            "prompts": APPLICATION_PROMPTS["score_prompts"],
                                                                            "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                                            },
                                                                    help="recommend summary")
                                qa_button = qa_col.button(":question: qa",
                                                            key=f"qa_{i}_pdf_{pdf_page_idx}",
                                                            on_click=enhance_answer,
                                                            kwargs={"usage": 'qa',
                                                                    "prompts": APPLICATION_PROMPTS["multimodal_qa"],
                                                                    "document_level_summary":st.session_state["pdf_generated_summary"][i][pdf_page_idx],
                                                                    },
                                                            help="qa summary")

                            st.divider()
                        process_bar.progress(num_selected / num_selected,
                                              text=f"{num_selected} PDF Completed")
                else:
                    st.error("Unable to get file, please upload file")





