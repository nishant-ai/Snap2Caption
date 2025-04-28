import React from 'react';
import {
  Box,
  Flex,
  Text,
  Button,
  Stack,
  Link,
  useColorModeValue,
  useBreakpointValue,
  Image,
} from '@chakra-ui/react';

export default function Navbar() {
  return (
    <Box>
      <Flex
        bg={useColorModeValue('white', 'gray.800')}
        color={useColorModeValue('gray.600', 'white')}
        minH={'60px'}
        py={{ base: 2 }}
        px={{ base: 4 }}
        borderBottom={1}
        borderStyle={'solid'}
        borderColor={useColorModeValue('gray.200', 'gray.900')}
        align={'center'}
        position="sticky"
        top="0"
        zIndex="1000"
        boxShadow="sm"
      >
        <Flex
          flex={{ base: 1 }}
          justify={{ base: 'center', md: 'start' }}
          align={'center'}
        >
          <Text
            textAlign={useBreakpointValue({ base: 'center', md: 'left' })}
            fontFamily={'heading'}
            color={useColorModeValue('teal.600', 'white')}
            fontWeight="bold"
            fontSize="xl"
          >
            Snap2Caption
          </Text>

          <Flex display={{ base: 'none', md: 'flex' }} ml={10}>
            <DesktopNav />
          </Flex>
        </Flex>

        {/* Removed Sign In/Sign Up buttons as requested */}
      </Flex>
    </Box>
  );
}

const DesktopNav = () => {
  const linkColor = useColorModeValue('gray.600', 'gray.200');
  const linkHoverColor = useColorModeValue('teal.800', 'white');

  return (
    <Stack direction={'row'} spacing={4}>
      {NAV_ITEMS.map((navItem) => (
        <Box key={navItem.label}>
          <Link
            p={2}
            href={navItem.href ?? '#'}
            fontSize={'sm'}
            fontWeight={500}
            color={linkColor}
            _hover={{
              textDecoration: 'none',
              color: linkHoverColor,
            }}
          >
            {navItem.label}
          </Link>
        </Box>
      ))}
    </Stack>
  );
};

const NAV_ITEMS = [
  {
    label: 'Home',
    href: '#',
  },
  {
    label: 'Features',
    href: '#features',
  },
  {
    label: 'How It Works',
    href: '#how-it-works',
  },
  {
    label: 'About',
    href: '#about',
  },
];
