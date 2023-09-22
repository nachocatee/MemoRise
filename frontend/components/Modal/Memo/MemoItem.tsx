// 라이브러리
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  Image,
  Pressable,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";

// 스타일
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import { TextInput } from "react-native-gesture-handler";
import Colors from "../../../constants/colors";

const MemoItem = () => {
  return (
    <>
      <View style={styles.memoContainer}>
        <LinearGradient
          colors={["#FFFFFF", "#F5F5F5"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
          style={{ flex: 1 }}
        >
          <View style={styles.innerContainer}>
            <View style={styles.calenderContainer}>
              <Text style={styles.calendar}>2023. 09. 22</Text>
            </View>
            <Text style={styles.content} numberOfLines={4} ellipsizeMode="tail">
              Hate to give the satisfaction, asking how you're doing nowHow's
              the castle built off people you pretend to care about?Just what
              you wantedLook at you, cool guy, you got it I see the parties and
              the
            </Text>
            <View style={styles.bottomContainer}>
              <Text style={styles.nickname}>권소정</Text>
              <Text style={styles.open}>전체공개</Text>
            </View>
          </View>
        </LinearGradient>
      </View>
      <Pressable>
        <Image
          source={require("../../../assets/icons/bookmarkblue.png")}
          style={styles.bookmarkSize}
        />
      </Pressable>
    </>
  );
};

export default MemoItem;

const styles = StyleSheet.create({
  memoContainer: {
    width: calculateDynamicWidth(306),
    minheight: calculateDynamicWidth(104),
    maxHeight: calculateDynamicWidth(185),
    borderRadius: calculateDynamicWidth(15),
    overflow: "scroll",
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(306) / 2 },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
  },

  innerContainer: {
    margin: calculateDynamicWidth(9),
  },

  bookmarkSize: {
    width: calculateDynamicWidth(17),
    height: calculateDynamicWidth(23),
    marginLeft: calculateDynamicWidth(50),
    bottom: calculateDynamicWidth(520),
    position: "absolute",
  },

  calenderContainer: {
    alignItems: "flex-end",
  },

  calendar: {
    color: Colors.hover,
    fontFamily: "Pretendard-Regular",
  },

  content: {
    color: Colors.text,
    fontSize: 18,
    fontFamily: "Pretendard-Regular",
    marginVertical: 5,
    marginHorizontal: 10,
  },

  bottomContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },

  nickname: {
    color: Colors.blue500,
    fontFamily: "Pretendard-Medium",
  },

  open: {
    color: "rgba(76, 106, 255, 0.6)",
    fontFamily: "Pretendard-Medium",
  },
});
