import { useState } from 'react';
import {
  Box,
  Heading,
  Button,
  Image,
  Input,
  Text,
  VStack,
  HStack,
} from '@chakra-ui/react';

function App() {
  const [image, setImage] = useState(null);
  const [caption, setCaption] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setImage(file);
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const uploadImage = async () => {
    if (!image) {
      alert('Please select an image first.');
      return;
    }
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://127.0.0.1:8000/generate-caption/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setCaption(data.caption || data.error || 'Error generating caption.');
    } catch (error) {
      console.error(error);
      setCaption('Failed to communicate with server.');
    }
  };

  const rateCaption = (feedback) => {
    if (feedback === 'up') {
      alert('Thank you for your positive feedback! 👍');
    } else {
      alert('Thanks for the feedback! We will improve! 👎');
    }
  };

  return (
    <Box textAlign="center" py={10} px={6} bg="gray.50" minH="100vh">
      <Heading as="h1" size="xl" mb={6}>
        🖼️ Image Caption Generator
      </Heading>

      <VStack spacing={5}>
        <Input type="file" accept="image/*" onChange={handleFileChange} width="auto" />

        {previewUrl && (
          <Image
            src={previewUrl}
            alt="Uploaded Preview"
            borderRadius="md"
            boxSize="300px"
            objectFit="cover"
          />
        )}

        <Button colorScheme="teal" onClick={uploadImage}>
          Generate Caption
        </Button>

        {caption && (
          <Text fontSize="xl" color="gray.700" mt={4}>
            📜 "{caption}"
          </Text>
        )}

        {caption && (
          <HStack spacing={10} pt={4}>
            <Button onClick={() => rateCaption('up')} fontSize="2xl">
              👍
            </Button>
            <Button onClick={() => rateCaption('down')} fontSize="2xl">
              👎
            </Button>
          </HStack>
        )}
      </VStack>
    </Box>
  );
}

export default App;
