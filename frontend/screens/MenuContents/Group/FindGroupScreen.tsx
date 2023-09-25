import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { View, Text } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import SmallBtn from "../../../components/Button/SmallBtn";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import SeachInput from "../../../components/SearchInput";
import { styles } from "./GroupStyle";

type RootStackParamList = {
  MakeGroup: undefined;
};

const FindGroupScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();
  const MakeGroupHandler = () => navigation.navigate("MakeGroup");

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>그룹 찾기</Text>
          <SmallBtn onPress={MakeGroupHandler}>만들기</SmallBtn>
        </View>
        <View style={styles.contentsContainer}>{/* <SeachInput /> */}</View>
      </View>
    </LinearGradient>
  );
};

export default FindGroupScreen;
