import streamlit as st
import os
import zipfile
import requests
import pandas as pd
import numpy as np
import json
from tcia_utils import nbia
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#st.markdown(
#    """
#    <style>
#    .stApp {
#        background-color: black;
#        color: white;
#    }
#    .stApp h1 {
#        color: white;
#    }
#    .stApp .markdown-text-container {
#        color: white;
#    }
#    .stSidebar {
#        background-color: black;
#        color: white;
#    }
#    </style>
#    """,
#    unsafe_allow_html=True
#)

# Function to download and cache the zip file
@st.cache_resource
def download_and_extract_zip():
    # URL of the zip file
    url = "https://www.cancerimagingarchive.net/wp-content/uploads/TOMPEI-CMMD_v01_20241220.zip"

    # Create a cache directory if it doesn't exist
    cache_dir = "cached_data"
    os.makedirs(cache_dir, exist_ok=True)

    # Path for the zip file and extraction directory
    zip_path = os.path.join(cache_dir, "TOMPEI-CMMD.zip")
    extract_path = os.path.join(cache_dir, "extracted")

    # Download the zip file if not already downloaded
    if not os.path.exists(zip_path):
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)

    # Extract the zip file if not already extracted
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    return extract_path

# Function to extract patient IDs from JSON files
def extract_patient_json_mapping(extract_path):
    # Dictionary to store full JSON file paths
    json_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(extract_path):
        for filename in files:
            # Explicitly filter out hidden files and ensure it's a JSON
            if not filename.startswith('.') and filename.endswith('.json'):
                json_files.append(os.path.join(root, filename))

    # Create a mapping of patient IDs to their JSON files
    patient_json_dict = {}
    for json_path in json_files:
        patient_id = os.path.basename(json_path)[:7]

        # If patient ID not in dict, create a new list
        if patient_id not in patient_json_dict:
            patient_json_dict[patient_id] = []

        # Add full path of the JSON file
        patient_json_dict[patient_id].append(json_path)

    return patient_json_dict

# Function to load annotation from JSON
def load_annotations(json_path):
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        annotation_data = []
        for annotation in annotations:
            if 'cgPoints' in annotation:
                points = annotation['cgPoints']
                x_coords = [point['x'] for point in points]
                y_coords = [point['y'] for point in points]

                # Get label and color, with defaults if not present
                label = annotation.get('label', 'Unknown').strip()
                color = annotation.get('color', '#FF0000')

                annotation_data.append({
                    'x_coords': x_coords,
                    'y_coords': y_coords,
                    'label': label,
                    'color': color
                })

        return annotation_data
    except Exception as e:
        st.warning(f"Error reading annotation file: {e}")
        return []

# Updated function to display DICOM image with multiple annotations
def display_dicom_with_annotation(dicom_path, annotation_path=None, show_annotations=None):
    # Read DICOM file
    dcm = pydicom.dcmread(dicom_path)

    # Convert to numpy array
    image = dcm.pixel_array

    # Create figure and plot image with consistent scaling
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use a fixed interpolation method to prevent rescaling
    im = ax.imshow(image, cmap='gray', interpolation='nearest')

    # Ensure consistent axis limits
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Flip y-axis to match DICOM orientation

    # Add annotations if available
    legend_elements = []
    if annotation_path and show_annotations:
        annotations = load_annotations(annotation_path)

        for annotation in annotations:
            label = annotation['label']
            # Only display annotation if its type is enabled
            if show_annotations.get(label, True):  # Default to True if not specified
                x_coords = annotation['x_coords']
                y_coords = annotation['y_coords']
                color = annotation['color']

                # Close the polygon by adding the first point at the end
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])

                # Plot polygon
                polygon = ax.plot(x_coords, y_coords, color=color, linewidth=2)[0]
                ax.fill(x_coords, y_coords, color=color, alpha=0.2)

            # Always add to legend elements, even if annotation is hidden
            legend_elements.append(patches.Patch(facecolor=annotation['color'],
                                              alpha=0.2,
                                              edgecolor=annotation['color'],
                                              label=label))

    # Always show legend if there are annotations
    if legend_elements:
        ax.legend(handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(0, 1), # Adjust bbox_to_anchor for alignment
            title="Annotation Type",
            fontsize='large') # Set the font size

    ax.axis('off')

    # Adjust layout to remove any padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    return fig

# check if dicom is MLO
def is_mlo_view(dcm):
    """
    Check if the DICOM image is an MLO view by examining the View Code Sequence.
    """
    try:
        # Check for View Code Sequence (0054,0220)
        if hasattr(dcm, 'ViewCodeSequence'):
            view_seq = dcm.ViewCodeSequence
            if view_seq and hasattr(view_seq[0], 'CodeMeaning'):
                code_meaning = view_seq[0].CodeMeaning.lower()
                return 'medio-lateral oblique' in code_meaning or 'mlo' in code_meaning
    except Exception as e:
        st.warning(f"Error checking MLO view: {e}")
    return False

# Main Streamlit app
def main():
    st.title("TOMPEI-CMMD Annotation Viewer")

    # Initialize session state
    if 'current_patient_index' not in st.session_state:
        st.session_state.current_patient_index = 0
    if 'patient_images' not in st.session_state:
        st.session_state.patient_images = {}
    if 'images_downloaded' not in st.session_state:
        st.session_state.images_downloaded = set()
    if 'annotation_toggles' not in st.session_state:
        st.session_state.annotation_toggles = {}

    # Download and extract zip file
    extract_path = download_and_extract_zip()

    # Fetch TCIA series metadata using cache
    @st.cache_data
    def get_series_data():
        return nbia.getSeries(collection='CMMD', format='df')

    series = get_series_data()

    # Extract patient IDs and their JSON files
    @st.cache_data
    def get_patient_mapping(_extract_path):
        return extract_patient_json_mapping(_extract_path)

    patient_json_dict = get_patient_mapping(extract_path)

    # Check if patient IDs were found
    if not patient_json_dict:
        st.error("No patient IDs could be extracted from JSON files")
        return

    # Sort patient IDs alphabetically
    sorted_patient_ids = sorted(list(patient_json_dict.keys()))

    # Sidebar controls
    st.sidebar.header("Select Patient")
    st.sidebar.write(f"Found {len(patient_json_dict)} patient IDs")

    # Navigation and Patient Selection
    col1, col2, col3 = st.sidebar.columns([2,1,2])

    def update_patient_index(new_index):
        st.session_state.current_patient_index = new_index

    # Select patient via dropdown
    selected_patient = st.sidebar.selectbox(
        "Choose Patient",
        sorted_patient_ids,
        index=st.session_state.current_patient_index,
        key="patient_selector",
        on_change=lambda: update_patient_index(sorted_patient_ids.index(st.session_state.patient_selector))
    )

    with col1:
        if st.button("◀ Previous"):
            update_patient_index(max(0, st.session_state.current_patient_index - 1))

    with col3:
        if st.button("Next ▶"):
            update_patient_index(min(len(sorted_patient_ids) - 1, st.session_state.current_patient_index + 1))

    # Annotation Controls Section
    st.sidebar.header("Annotation Controls")

    # Initialize annotation toggles for the current patient
    if selected_patient not in st.session_state.annotation_toggles:
        st.session_state.annotation_toggles[selected_patient] = {}

    # Download and process images for the selected patient
    if selected_patient not in st.session_state.images_downloaded:
        with st.spinner(f"Loading images for patient {selected_patient}..."):
            matching_series = series[series['PatientID'] == selected_patient]

            if not matching_series.empty:
                try:
                    series_uid_list = matching_series['SeriesInstanceUID'].tolist()
                    downloadPath = os.path.join("images", selected_patient)
                    os.makedirs(downloadPath, exist_ok=True)

                    # Download series if not already present
                    if not os.path.exists(downloadPath) or not os.listdir(downloadPath):
                        nbia.downloadSeries(series_uid_list, input_type='list', path=downloadPath)

                    mlo_images = []
                    mlo_annotations = []
                    patient_json_files = patient_json_dict[selected_patient]

                    # Find and process DICOM files
                    for root, dirs, files in os.walk(downloadPath):
                        for file in files:
                            if file.endswith('.dcm'):
                                dcm_path = os.path.join(root, file)
                                try:
                                    dcm = pydicom.dcmread(dcm_path)

                                    # Debug info commented out
                                    if hasattr(dcm, 'ViewCodeSequence'):
                                        view_seq = dcm.ViewCodeSequence
                                    #    if view_seq and hasattr(view_seq[0], 'CodeMeaning'):
                                    #        st.write(f"Found view {view_seq[0].CodeMeaning}: in {dcm_path}")

                                    if is_mlo_view(dcm):
                                    #    st.write(f"Found MLO image: {dcm_path}")  # Debug info
                                        matching_annotations = [
                                            json_file for json_file in patient_json_files
                                            if selected_patient == os.path.basename(json_file)[:7]
                                        ]

                                        if matching_annotations:
                                            mlo_images.append(dcm_path)
                                            mlo_annotations.append(matching_annotations[0])
                                    #        st.write(f"Matched with annotation: {matching_annotations[0]}")
                                except Exception as e:
                                    st.warning(f"Error reading DICOM file {file}: {e}")

                    # Store images for this patient
                    st.session_state.patient_images[selected_patient] = {
                        'images': mlo_images,
                        'annotations': mlo_annotations
                    }
                    st.session_state.images_downloaded.add(selected_patient)

                except Exception as e:
                    st.error(f"Error processing patient {selected_patient}: {e}")

    # Display results
    patient_images = st.session_state.patient_images.get(selected_patient, {'images': [], 'annotations': []})

    if len(patient_images['images']) > 0:
        # debug step
        #st.success(f"Found {len(patient_images['images'])} MLO images")

        # Get unique annotation types for this patient
        annotation_types = set()
        for json_path in patient_images['annotations']:
            annotations = load_annotations(json_path)
            for annotation in annotations:
                annotation_types.add(annotation['label'])

        # Create toggles for each annotation type
        annotation_states = {}
        for annotation_type in annotation_types:
            if annotation_type not in st.session_state.annotation_toggles[selected_patient]:
                st.session_state.annotation_toggles[selected_patient][annotation_type] = True

            annotation_states[annotation_type] = st.sidebar.checkbox(
                f"Show {annotation_type}",
                value=st.session_state.annotation_toggles[selected_patient][annotation_type],
                key=f"toggle_{selected_patient}_{annotation_type}"
            )
            st.session_state.annotation_toggles[selected_patient][annotation_type] = annotation_states[annotation_type]

        # Display images with annotations
        for dcm_path, json_path in zip(patient_images['images'], patient_images['annotations']):
            st.markdown(f"### Patient: {selected_patient}")
            st.markdown(f"Annotation: {os.path.basename(json_path)}")
            #st.markdown(f"Series UID / Image: {dcm_path}")

            # Create figure with current annotation states
            figure_key = f"fig_{dcm_path}_{hash(str(annotation_states))}"
            if figure_key not in st.session_state:
                st.session_state[figure_key] = display_dicom_with_annotation(
                    dcm_path,
                    json_path,
                    annotation_states
                )

            # Display the cached figure
            st.pyplot(st.session_state[figure_key])
    else:
        st.warning(f"No MLO images found for patient {selected_patient}")

if __name__ == "__main__":
    main()
