package com.tjjhtjh.memorise.domain.user.service.dto.request;

import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Data
@Getter
@NoArgsConstructor
public class JoinRequest {
    private String email;
    private String nickname;
    private String profile;
}
