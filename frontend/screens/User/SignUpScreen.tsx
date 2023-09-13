import { View, Text } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import CancelHeaderHighlight from "../../components/Header/CancelHeaderHighlight";

const SignUpScreen = () => {
  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <View>
        <CancelHeaderHighlight />
      </View>
    </LinearGradient>
  );
};

export default SignUpScreen;
