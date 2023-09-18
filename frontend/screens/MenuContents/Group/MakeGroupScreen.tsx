import { View, Text, Image } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { styles } from "./GroupStyle";

const MakeGroupScreen = () => {
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
        <View style={styles.titleContainer}>
          <Text style={styles.title}>그룹 만들기</Text>
        </View>
      </View>
    </LinearGradient>
  );
};

export default MakeGroupScreen;
