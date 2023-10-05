package com.tjjhtjh.memorise.domain.team.repository.entity;

import com.tjjhtjh.memorise.global.entity.BaseEntity;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;

@DynamicUpdate
@DynamicInsert
@Getter
@Entity
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Team extends BaseEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "team_seq")
    private Long teamSeq;

    private String name;
    private Long owner;
    private String password;

    @Builder
    public Team(String name, Long owner) {
        this.name = name;
        this.owner = owner;
    }

    @Builder
    public Team(String name, Long owner, String password) {
        this.name = name;
        this.owner = owner;
        this.password = password;
    }

    public void update(String name) {
        this.name = name;
    }

    public void update(String name, String password) {
        this.name = name;
        this.password = password;
    }
}
