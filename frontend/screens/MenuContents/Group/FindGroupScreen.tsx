import { View, Text, Image } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { styles } from "./GroupStyle";

const FindGroupScreen = () => {
  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View>
        <Text style={styles.title}>그룹 찾기</Text>
      </View>
    </LinearGradient>
  );
};

export default FindGroupScreen;
