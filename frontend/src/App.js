// import { useState } from 'react';
// import {
//   Box,
//   Heading,
//   Button,
//   Image,
//   Input,
//   Text,
//   VStack,
//   HStack,
//   Container,
//   SimpleGrid,
//   Icon,
//   Flex,
//   Stack,
//   useColorModeValue,
// } from '@chakra-ui/react';
// import { FaImage, FaRobot, FaThumbsUp } from 'react-icons/fa';
// import Navbar from './components/Navbar';
// import config from './config';


// function Feature({ title, text, icon }) {
//   return (
//     <Stack align={'center'} textAlign={'center'}>
//       <Flex
//         w={16}
//         h={16}
//         align={'center'}
//         justify={'center'}
//         color={'white'}
//         rounded={'full'}
//         bg={'teal.500'}
//         mb={1}
//       >
//         {icon}
//       </Flex>
//       <Text fontWeight={600}>{title}</Text>
//       <Text color={'gray.600'}>{text}</Text>
//     </Stack>
//   );
// }

// function App() {
//   const [image, setImage] = useState(null);
//   const [caption, setCaption] = useState('');
//   const [previewUrl, setPreviewUrl] = useState('');
//   const [isCaptionValid, setIsCaptionValid] = useState(false); // ✅ new state

//   const handleFileChange = (event) => {
//     const file = event.target.files[0];
//     setImage(file);
//     if (file) {
//       const reader = new FileReader();
//       reader.onloadend = () => {
//         setPreviewUrl(reader.result); // base64 string with prefix
//       };
//       reader.readAsDataURL(file);
//     }
//   };

//   const uploadImage = async () => {
//     if (!image) {
//       alert('Please select an image first.');
//       return;
//     }

//     const reader = new FileReader();

//     reader.onloadend = async () => {
//       const base64Image = reader.result.split(',')[1]; // remove prefix

//       const payload = {
//         image: base64Image,
//         prompt:
//           'You are a social media influencer. Write a captivating Instagram caption for this image.',
//       };

//       try {
//         const response = await fetch(`${config.API_BASE_URL}/generate_caption`, {
//           method: 'POST',
//           headers: {
//             'Content-Type': 'application/json',
//           },
//           body: JSON.stringify(payload),
//         });

//         const data = await response.json();
//         console.log('Response:', data);
//         const rawCaption = data.caption;
//         const parts = rawCaption?.split('[/INST]');
//         const cleanedCaption = parts?.length > 1 ? parts[1].trim() : rawCaption;

//         setCaption(cleanedCaption || data.error || 'Error generating caption.');
//         setIsCaptionValid(!!cleanedCaption); // ✅ only if valid caption
//       } catch (error) {
//         console.error(error);
//         setCaption('Failed to communicate with server.');
//         setIsCaptionValid(false); // invalidate
//       }
//     };

//     reader.readAsDataURL(image);
//   };

//   const rateCaption = (feedback) => {
//     if (feedback === 'up') {
//       alert('Thank you for your positive feedback! 👍');
//     } else {
//       const reason = prompt('We’re sorry to hear that. Could you tell us what was wrong?');
//       if (reason && reason.trim() !== '') {
//         const rawBase64 = previewUrl.split(',')[1]; // remove data:image/jpeg;base64,
//         console.log('🖼️ Image (Base64):', rawBase64);
//         console.log('📜 Original Caption:', caption);
//         console.log('📝 Feedback:', reason);
//       } else {
//         console.log('👎 User gave negative feedback but didn’t leave a comment.');
//       }
//       alert('Thanks for the feedback! We will improve! 👎');
//     }
//   };

//   return (
//     <Box bg={useColorModeValue('gray.50', 'gray.900')} minH="100vh">
//       <Navbar />

//       {/* Hero Section */}
//       <Box bg={useColorModeValue('teal.50', 'gray.800')} py={20} px={8} textAlign="center">
//         <Container maxW={'container.xl'}>
//           <Heading
//             fontWeight={700}
//             fontSize={{ base: '3xl', sm: '4xl', md: '5xl' }}
//             lineHeight={'110%'}
//             color={'teal.600'}
//             mb={6}
//           >
//             Transform Images into Words <br />
//             <Text as={'span'} color={'teal.400'}>
//               with Snap2Caption
//             </Text>
//           </Heading>
//           <Text color={'gray.600'} maxW={'3xl'} mx={'auto'} fontSize={'xl'} mb={10}>
//             Our advanced AI technology analyzes your images and generates accurate, detailed captions in seconds. Perfect for content creators, social media managers, and accessibility needs.
//           </Text>
//         </Container>
//       </Box>

//       {/* Main Section */}
//       <Box id="how-it-works" py={10} px={6} textAlign="center">
//         <Container maxW={'container.lg'}>
//           <Heading as="h2" size="xl" mb={10} color={'teal.600'}>
//             🎨 🖼️ 🖌️ Create Caption for your image 🎭 📸 🎬
//           </Heading>

//           <VStack spacing={8} p={6} bg={useColorModeValue('white', 'gray.700')} borderRadius="lg" boxShadow="md">
//             <Text fontSize="lg" color={'gray.600'}>
//               Upload an image and let our AI generate a descriptive caption for you.
//             </Text>

//             <Button
//               as="label"
//               htmlFor="file-upload"
//               colorScheme="teal"
//               variant="outline"
//               leftIcon={<Icon as={FaImage} />}
//               size="lg"
//               px={8}
//               cursor="pointer"
//               borderWidth="2px"
//               borderRadius="md"
//               borderStyle="dashed"
//               _hover={{ bg: 'teal.50', borderColor: 'teal.300' }}
//             >
//               Choose an Image
//               <Input
//                 id="file-upload"
//                 type="file"
//                 accept="image/*"
//                 onChange={handleFileChange}
//                 width="auto"
//                 display="none"
//               />
//             </Button>

//             {previewUrl && (
//               <Image
//                 src={previewUrl}
//                 alt="Uploaded Preview"
//                 borderRadius="md"
//                 boxSize="300px"
//                 objectFit="cover"
//               />
//             )}

//             <Button
//               colorScheme="teal"
//               size="lg"
//               onClick={uploadImage}
//               px={8}
//               leftIcon={<Icon as={FaRobot} />}
//               boxShadow="md"
//               _hover={{ transform: 'translateY(-2px)', boxShadow: 'lg' }}
//               _active={{ transform: 'translateY(0)', boxShadow: 'md' }}
//               transition="all 0.2s"
//             >
//               Generate Caption
//             </Button>

//             {caption && (
//               <Box p={4} bg="teal.50" borderRadius="md" width="full">
//                 <Text fontSize="xl" color="gray.700" fontWeight="medium">
//                   📜 "{caption}"
//                 </Text>
//               </Box>
//             )}

//             {isCaptionValid && (
//               <Box pt={4}>
//                 <Text fontSize="md" color="gray.600" mb={3}>
//                   How was this caption?
//                 </Text>
//                 <HStack spacing={6} justify="center">
//                   <Button
//                     onClick={() => rateCaption('up')}
//                     colorScheme="green"
//                     variant="outline"
//                     size="md"
//                     leftIcon={<Text fontSize="xl">👍</Text>}
//                     _hover={{ bg: 'green.50', transform: 'scale(1.05)' }}
//                     transition="all 0.2s"
//                   >
//                     Good
//                   </Button>
//                   <Button
//                     onClick={() => rateCaption('down')}
//                     colorScheme="red"
//                     variant="outline"
//                     size="md"
//                     leftIcon={<Text fontSize="xl">👎</Text>}
//                     _hover={{ bg: 'red.50', transform: 'scale(1.05)' }}
//                     transition="all 0.2s"
//                   >
//                     Needs Improvement
//                   </Button>
//                 </HStack>
//               </Box>
//             )}
//           </VStack>
//         </Container>
//       </Box>

//       {/* Features Section */}
//       <Box id="features" py={20}>
//         <Container maxW={'container.xl'}>
//           <Heading textAlign={'center'} fontSize={'3xl'} py={10} fontWeight={'bold'} color={'teal.600'}>
//             Why Choose Snap2Caption?
//           </Heading>
//           <SimpleGrid columns={{ base: 1, md: 3 }} spacing={10} px={10}>
//             <Feature
//               icon={<Icon as={FaImage} w={10} h={10} />}
//               title={'Instant Processing'}
//               text={'Upload any image and get a detailed caption within seconds.'}
//             />
//             <Feature
//               icon={<Icon as={FaRobot} w={10} h={10} />}
//               title={'AI-Powered Accuracy'}
//               text={'State-of-the-art models ensure relevant, context-aware captions.'}
//             />
//             <Feature
//               icon={<Icon as={FaThumbsUp} w={10} h={10} />}
//               title={'Continuous Improvement'}
//               text={'We learn from your feedback to improve caption quality over time.'}
//             />
//           </SimpleGrid>
//         </Container>
//       </Box>

//       {/* About Section */}
//       <Box id="about" bg={useColorModeValue('teal.50', 'gray.800')} py={20} px={8}>
//         <Container maxW={'container.lg'} textAlign="center">
//           <Heading as="h2" size="xl" mb={6} color={'teal.600'}>
//             About Snap2Caption
//           </Heading>
//           <Text fontSize="lg" color={'gray.600'} maxW={'3xl'} mx={'auto'} mb={10}>
//             Snap2Caption bridges the gap between images and words. We make visual content more accessible, searchable, and usable — for creators, marketers, and those who need visual assistance.
//           </Text>
//         </Container>
//       </Box>
//     </Box>
//   );
// }

// export default App;


import { useState } from 'react';
import {
  Box, Heading, Button, Image, Input, Text, VStack, HStack, Container,
  SimpleGrid, Icon, Flex, Stack, useColorModeValue
} from '@chakra-ui/react';
import { FaImage, FaRobot, FaThumbsUp } from 'react-icons/fa';
import Navbar from './components/Navbar';
import { v4 as uuidv4 } from 'uuid';
import config from './config'; // Contains API_BASE_URL

function App() {
  const [image, setImage] = useState(null);
  const [caption, setCaption] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [isCaptionValid, setIsCaptionValid] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setImage(file);
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const uploadImage = async () => {
    if (!image) {
      alert('Please select an image first.');
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Image = reader.result.split(',')[1];
      const payload = {
        image: base64Image,
        prompt:
          'You are a social media influencer. Write a captivating Instagram caption for this image.',
      };

      try {
        const response = await fetch(`${config.Caption_API_BASE_URL}/generate_caption`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        const data = await response.json();
        const parts = data.caption?.split('[/INST]');
        const cleaned = parts?.length > 1 ? parts[1].trim() : data.caption;

        setCaption(cleaned || data.error || 'Error generating caption.');
        setIsCaptionValid(!!cleaned);
      } catch (error) {
        console.error(error);
        setCaption('Failed to communicate with server.');
        setIsCaptionValid(false);
      }
    };
    reader.readAsDataURL(image);
  };

  const rateCaption = async (feedbackType) => {
    if (feedbackType === 'up') {
      alert('Thank you for your positive feedback! 👍');
    } else {
      const reason = prompt('We’re sorry to hear that. Could you tell us what was wrong?');
      if (reason && reason.trim() !== '') {
        const uid = uuidv4();
        const rawBase64 = previewUrl.split(',')[1];

        const payload = {
          uid,
          image: rawBase64,
          caption,
          feedback: reason,
        };

        try {
          await fetch(`${config.Feedback_API_BASE_URL}/store_feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          alert('Thanks for the feedback! 👎');
        } catch (error) {
          console.error('Failed to store feedback:', error);
          alert('Something went wrong while saving feedback.');
        }
      } else {
        console.log('👎 User gave negative feedback but didn’t leave a comment.');
      }
    }
  };

  return (
    <Box bg={useColorModeValue('gray.50', 'gray.900')} minH="100vh">
      <Navbar />
      <Box py={20} px={8} textAlign="center" bg={useColorModeValue('teal.50', 'gray.800')}>
        <Container maxW="container.xl">
          <Heading fontWeight={700} fontSize={{ base: '3xl', sm: '4xl', md: '5xl' }} color="teal.600">
            Transform Images into Words <br />
            <Text as="span" color="teal.400">with Snap2Caption</Text>
          </Heading>
        </Container>
      </Box>

      <Box py={10} px={6} textAlign="center">
        <Container maxW="container.lg">
          <VStack spacing={8} p={6} bg={useColorModeValue('white', 'gray.700')} borderRadius="lg" boxShadow="md">
            <Text>Upload an image and let our AI generate a descriptive caption for you.</Text>
            <Button as="label" htmlFor="file-upload" colorScheme="teal" variant="outline" leftIcon={<Icon as={FaImage} />}>
              Choose Image
              <Input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} hidden />
            </Button>

            {previewUrl && (
              <Image src={previewUrl} alt="Uploaded Preview" borderRadius="md" boxSize="300px" objectFit="cover" />
            )}

            <Button onClick={uploadImage} colorScheme="teal" leftIcon={<Icon as={FaRobot} />}>
              Generate Caption
            </Button>

            {caption && (
              <Box p={4} bg="teal.50" borderRadius="md" width="full">
                <Text fontSize="xl" color="gray.700" fontWeight="medium">
                  📜 "{caption}"
                </Text>
              </Box>
            )}

            {isCaptionValid && (
              <Box pt={4}>
                <Text fontSize="md" color="gray.600" mb={3}>How was this caption?</Text>
                <HStack spacing={6} justify="center">
                  <Button onClick={() => rateCaption('up')} colorScheme="green" variant="outline">👍 Good</Button>
                  <Button onClick={() => rateCaption('down')} colorScheme="red" variant="outline">👎 Needs Improvement</Button>
                </HStack>
              </Box>
            )}
          </VStack>
        </Container>
      </Box>
    </Box>
  );
}

export default App;

