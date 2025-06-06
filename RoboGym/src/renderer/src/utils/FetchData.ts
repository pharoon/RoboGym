export const GetModels = async () => {
  try {
    const response = await fetch("http://localhost:5000/models");

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const models = await response.json();
    console.warn("modes are now ", models)
    return models;
  } catch (e) {
    console.error("❌ Error occurred while fetching models:", e);
    return []; // Return empty array or `null` to avoid undefined usage
  }
};
